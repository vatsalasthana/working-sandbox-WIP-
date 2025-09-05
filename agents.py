import asyncio
import json
from typing import Dict, List, Any, Optional, Annotated
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import requests
from tools import tool_executor
from prompts import prompts_manager
from utils import logger
from database import db_handler


class AgentState(TypedDict):
    """State shared between agents"""
    messages: Annotated[List[Any], add_messages]
    user_query: str
    available_files: List[Dict]
    tool_results: List[Dict]
    final_response: str


# Define available tools (same as before)
@tool
def search_documents(query: str, file_ids: List[str] = None) -> Dict[str, Any]:
    """Search through uploaded documents for relevant information"""
    payload = {
        "query": query,
        "collection": "sandbox",
        "top_k": 10,
    }
    url = 'https://3ce4781230c0.ngrok-free.app/api/v1/chat/vector-search'
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            print("Rag data", data)
            return data.get("top_chunks", [])
        else:
            raise Exception(f"RAG API error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching RAG context: {e}")
        return []





@tool
async def web_search(query: str, num_results: int = 5, search_type: str = "search") -> Dict[str, Any]:
    """Search the web using Serper.dev API for current information"""
    try:
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            return {"status": "error", "error": "SERPER_API_KEY not found", "source": "web_search"}
        
        url = "https://google.serper.dev/search"
        if search_type == "news":
            url = "https://google.serper.dev/news"
        
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(num_results, 10)}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if search_type == "search":
                organic_results = data.get("organic", [])
                for result in organic_results[:num_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "source": result.get("source", "")
                    })
            elif search_type == "news":
                news_results = data.get("news", [])
                for result in news_results[:num_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", "")
                    })
            
            context = "\n\n".join([
                f"**{result['title']}**\n{result['snippet']}\nSource: {result.get('source', 'Unknown')}"
                for result in results[:3] if result.get('snippet')
            ])
            
            return {
                "status": "success",
                "query": query,
                "search_type": search_type,
                "results": results,
                "context": context,
                "total_results": len(results),
                "source": "web_search"
            }
            
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {"status": "error", "error": str(e), "source": "web_search"}


available_tools = [search_documents, web_search]
tool_node = ToolNode(available_tools)


class SupervisorAgent:
    """Step 1: Decides what tools to use (ONE TIME ONLY)"""
    
    def __init__(self, st):
        self.st = st
        self.llm = self._get_supervisor_llm()
        self.llm_with_tools = self.llm.bind_tools(available_tools)
    
    def _get_supervisor_llm(self):
        provider = self.st.session_state.get('supervisor_provider', 'openai')
        
        if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif provider == 'openrouter' and os.getenv('OPENROUTER_API_KEY'):
            return ChatOpenAI(
                model="meta-llama/llama-guard-4-12b",
                temperature=0.7,
                api_key=os.getenv('OPENROUTER_API_KEY'),
                base_url='https://openrouter.ai/api/v1'
            )
        else:
            raise ValueError(f"No API key or unsupported provider: {provider}")
    
    async def process(self, state: AgentState) -> AgentState:
        """Decide what tools to use - called only once"""
        try:
            files_context = ""
            if state['available_files']:
                files_context = f"\n\nAvailable files:\n" + "\n".join([
                    f"- {f.get('filename', 'Unknown')} ({f.get('file_type', 'Unknown type')})"
                    for f in state['available_files']
                ])
            
            messages = [
                SystemMessage(content="""You are a Supervisor Agent. Your job is to decide which tools to call to answer the user's query.

Available tools:
- search_documents: Search uploaded documents
- web_search: Search the web (use search_type="news" for recent events)
- analyze_data: Analyze data
- get_file_info: Get file information

Choose the most appropriate tools based on the query. You will be called only ONCE, so choose wisely."""),
                
                HumanMessage(content=f"User query: {state['user_query']}{files_context}\n\nDecide which tools to use to best answer this query.")
            ]
            
            response = await self.llm_with_tools.ainvoke(messages)
            state["messages"].append(response)
            print(response)
            return state
            
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            state["messages"].append(AIMessage(content=f"Supervisor error: {str(e)}"))
            return state


class FinalResponderAgent:
    """Step 3: Generate final response"""
    
    def __init__(self, st):
        self.st = st
        self.llm = self._get_responder_llm()
    
    def _get_responder_llm(self):
        provider = self.st.session_state.get('responder_provider', 'openai')
        
        if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif provider == 'openrouter' and os.getenv('OPENROUTER_API_KEY'):
            return ChatOpenAI(
                model="meta-llama/llama-guard-4-12b",
                temperature=0.7,
                api_key=os.getenv('OPENROUTER_API_KEY'),
                base_url='https://openrouter.ai/api/v1'
            )
        else:
            raise ValueError(f"No API key or unsupported provider: {provider}")
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate final response based on tool results"""
        try:
            # Extract tool results from messages
            tool_results = []
            for msg in state["messages"]:
                if hasattr(msg, 'content') and msg.content and msg.content.startswith('{"status"'):
                    try:
                        result = json.loads(msg.content)
                        tool_results.append(result)
                    except:
                        pass
            
            # Prepare context
            context_parts = [f"Original Query: {state['user_query']}"]
            
            for result in tool_results:
                source = result.get('source', 'unknown')
                status = result.get('status', 'unknown')
                
                context_parts.append(f"\n--- {source.title()} Result ({status}) ---")
                
                if status == "success":
                    if source == "web_search" and result.get('context'):
                        context_parts.append(f"Web Search Context:\n{result['context']}")
                    elif source == "documents" and result.get('context'):
                        context_parts.append(f"Document Context:\n{result['context']}")
                    elif source == "analysis" and result.get('analysis'):
                        context_parts.append(f"Analysis:\n{result['analysis']}")
                    elif source == "file_system":
                        files = result.get('files', [])
                        context_parts.append(f"Files: {json.dumps(files, indent=2)}")
                else:
                    context_parts.append(f"Error: {result.get('error', 'Unknown error')}")
            
            context = "\n".join(context_parts)
            
            messages = [
                SystemMessage(content="""You are the Final Responder Agent. Create a comprehensive, helpful response based on the tool results.

Guidelines:
- Address the user's query directly
- Use information from successful tool results
- Distinguish between web sources and document sources
- Be conversational and helpful
- Provide actionable insights when possible"""),
                
                HumanMessage(content=f"""Based on the following information, provide a comprehensive response:

{context}

Create a helpful response that directly addresses the user's query.""")
            ]
            
            response = await self.llm.ainvoke(messages)
            print(f"Final response: {response}")
            final_response = response.content if hasattr(response, 'content') else str(response)
            
            state["final_response"] = final_response
            state["tool_results"] = tool_results
            
            return state
            
        except Exception as e:
            logger.error(f"Final responder error: {e}")
            state["final_response"] = f"I apologize, but I encountered an error: {str(e)}"
            return state


class LinearOrchestrator:
    """Linear 3-step orchestrator: Supervisor -> Tools -> Final Responder"""
    
    def __init__(self, st):
        self.st = st
        self.supervisor = SupervisorAgent(st)
        self.final_responder = FinalResponderAgent(st)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build linear workflow: supervisor -> tools -> final_responder"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self.supervisor.process)
        workflow.add_node("tools", tool_node)
        workflow.add_node("final_responder", self.final_responder.process)
        
        # Linear flow: START -> supervisor -> tools -> final_responder -> END
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "tools")
        workflow.add_edge("tools", "final_responder")
        workflow.add_edge("final_responder", END)
        
        return workflow.compile()
    
    async def process_query(self, user_query: str, available_files: List[Dict] = None) -> Dict[str, Any]:
        """Process query through linear 3-step workflow"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            initial_state = AgentState(
                messages=[],
                user_query=user_query,
                available_files=available_files or [],
                tool_results=[],
                final_response=""
            )
            
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                'status': 'success',
                'final_response': final_state["final_response"],
                'tool_results': final_state["tool_results"]
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                'status': 'error',
                'final_response': f"I encountered an error: {str(e)}",
                'error': str(e)
            }


# Usage example:
async def main():
    """Example usage"""
    import streamlit as st
    
    orchestrator = LinearOrchestrator(st)
    
    # Test with web search
    result = await orchestrator.process_query(
        "What are the latest developments in AI technology this week?",
        available_files=[]
    )
    
    print("Web Search Example:")
    print(result['final_response'])
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())