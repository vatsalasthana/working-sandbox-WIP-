import asyncio
import json
from typing import Dict, List, Any, Optional, Annotated
import os
import httpx
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
import requests
import uuid
# from tools import tool_executor
from prompts import prompts_manager
from utils import logger
# from database import db_handler
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class AgentState(TypedDict):
    """State shared between agents"""
    messages: Annotated[List[Any], add_messages]
    user_query: str
    available_files: List[Dict]
    tool_results: List[Dict]
    final_response: str
    conversation_summary: Optional[str]
    session_id: str


# [Keep the same tool definitions - search_documents and web_search]
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
        )

        if response.status_code == 200:
            data = response.json()
            print("RAG data", data)
            chunks = data.get("top_chunks", [])
            
            if chunks:
                context = "\n\n".join([
                    f"**Document: {chunk.get('metadata', {}).get('filename', 'Unknown')}**\n{chunk.get('content', '')}"
                    for chunk in chunks[:3] if chunk.get('content')
                ])
                
                return {
                    "status": "success",
                    "query": query,
                    "results": chunks,
                    "context": context,
                    "total_results": len(chunks),
                    "source": "search_documents"
                }
            else:
                return {
                    "status": "success",
                    "query": query,
                    "results": [],
                    "context": "No relevant documents found.",
                    "total_results": 0,
                    "source": "search_documents"
                }
        else:
            raise Exception(f"RAG API error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching RAG context: {e}")
        return {
            "status": "error",
            "error": str(e),
            "source": "search_documents"
        }


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


class ConversationSummarizerAgent:
    """Agent to summarize conversation history when it gets too long"""
    
    def __init__(self, st):
        self.st = st
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    async def summarize_conversation(self, messages: List[Any]) -> str:
        """Summarize the conversation history"""
        try:
            conversation_messages = [
                msg for msg in messages 
                if isinstance(msg, (HumanMessage, AIMessage)) 
                and hasattr(msg, 'content') 
                and msg.content.strip()
            ]
            
            if not conversation_messages:
                return ""
            
            conversation_text = "\n\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in conversation_messages[-10:]
            ])
            
            summary_prompt = f"""Summarize the following conversation, focusing on:
1. Key information about the user (name, preferences, context)
2. Topics discussed and user interests
3. Important facts or details that should be remembered
4. Any ongoing context or tasks

Conversation:
{conversation_text}

Provide a concise summary that preserves important context:"""
            
            response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Conversation summary error: {e}")
            return "Previous conversation occurred but summary unavailable."


def should_use_tools(user_query: str, conversation_context: str = "") -> bool:
    """Determine if the query needs external tools or can be answered from conversation context"""
    # Queries that can be answered from conversation context
    context_queries = [
        "what is my name", "who am i", "what did i tell you", "what did we discuss",
        "remind me", "what was my", "do you remember", "what did i say",
        "tell me about myself", "what do you know about me"
    ]
    
    query_lower = user_query.lower()
    
    # If it's a context query and we have conversation context, don't use tools
    if any(phrase in query_lower for phrase in context_queries) and conversation_context:
        return False
    
    # If query asks for recent/current information, use web search
    if any(word in query_lower for word in ["latest", "recent", "current", "today", "news", "update"]):
        return True
    
    # If query mentions documents or files, use document search
    if any(word in query_lower for word in ["document", "file", "pdf", "report", "analyze"]):
        return True
    
    # For general knowledge or conversation-based queries, don't use tools
    return False


class SupervisorAgent:
    """Enhanced Supervisor that better decides when to use tools"""
    
    def __init__(self, st):
        self.st = st
        self.llm = self._get_supervisor_llm()
        self.llm_with_tools = self.llm.bind_tools(available_tools)
        self.summarizer = ConversationSummarizerAgent(st)
    
    def _get_supervisor_llm(self):
        provider = getattr(self.st, 'session_state', {}).get('supervisor_provider', 'openai')
        
        if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=os.getenv('OPENAI_API_KEY')
            )
    
    async def process(self, state: AgentState) -> AgentState:
        """Decide what tools to use based on query type and conversation context"""
        try:
            # Handle conversation memory and summarization
            conversation_context = ""
            if state.get('conversation_summary'):
                conversation_context = state['conversation_summary']
            
            if len(state['messages']) > 20:
                summary = await self.summarizer.summarize_conversation(state['messages'])
                state['conversation_summary'] = summary
                conversation_context = summary
            
            # Extract recent conversation for context
            recent_conversation = ""
            human_ai_messages = [
                msg for msg in state['messages'][-10:]
                if isinstance(msg, (HumanMessage, AIMessage)) 
                and hasattr(msg, 'content') 
                and msg.content.strip()
            ]
            
            if human_ai_messages:
                recent_conversation = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in human_ai_messages[-6:]  # Last 3 exchanges
                ])
            
            full_context = f"{conversation_context}\n\nRecent conversation:\n{recent_conversation}".strip()
            
            # Decide if tools are needed
            needs_tools = should_use_tools(state['user_query'], full_context)
            
            if not needs_tools:
                # Skip tools - add a message indicating no tools needed
                print(f"Supervisor: No tools needed for query: {state['user_query']}")
                state["messages"].append(AIMessage(content="NO_TOOLS_NEEDED"))
                return state
            
            # If tools are needed, proceed with tool selection
            files_context = ""
            if state['available_files']:
                files_context = f"\n\nAvailable files:\n" + "\n".join([
                    f"- {f.get('filename', 'Unknown')} ({f.get('file_type', 'Unknown type')})"
                    for f in state['available_files']
                ])
            
            messages = [
                SystemMessage(content=f"""You are a Supervisor Agent. Analyze the user's query and decide which tools to use.

Available tools:
- search_documents: Search uploaded documents
- web_search: Search the web (use search_type="news" for recent events)

Current conversation context:
{full_context}

Only use tools if the query requires external information that cannot be answered from the conversation context."""),
                
                HumanMessage(content=f"User query: {state['user_query']}{files_context}\n\nDecide which tools to use, or respond with no tools if the answer is in our conversation context.")
            ]
            
            response = await self.llm_with_tools.ainvoke(messages)
            state["messages"].append(response)
            print(f"Supervisor response with tools: {response}")
            return state
            
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            state["messages"].append(AIMessage(content=f"Supervisor error: {str(e)}"))
            return state


class FinalResponderAgent:
    """Enhanced Final Responder that prioritizes conversation context"""
    
    def __init__(self, st):
        self.st = st
        self.llm = self._get_responder_llm()
    
    def _get_responder_llm(self):
        provider = getattr(self.st, 'session_state', {}).get('responder_provider', 'openai')
        
        if provider == 'openai' and os.getenv('OPENAI_API_KEY'):
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv('OPENAI_API_KEY')
            )
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate response prioritizing conversation context"""
        try:
            # Check if supervisor decided no tools were needed
            no_tools_needed = any(
                isinstance(msg, AIMessage) and msg.content == "NO_TOOLS_NEEDED"
                for msg in state["messages"]
            )
            
            # Extract conversation context
            conversation_summary = state.get('conversation_summary', '')
            
            # Get recent conversation history
            recent_messages = [
                msg for msg in state['messages']
                if isinstance(msg, (HumanMessage, AIMessage))
                and hasattr(msg, 'content')
                and msg.content.strip()
                and msg.content != "NO_TOOLS_NEEDED"
            ]
            
            conversation_history = ""
            if recent_messages:
                conversation_history = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in recent_messages[-8:]  # Last 4 exchanges
                ])
            
            # Extract tool results if any
            tool_results = []
            if not no_tools_needed:
                for msg in state["messages"]:
                    if isinstance(msg, ToolMessage):
                        try:
                            if isinstance(msg.content, str):
                                try:
                                    result = json.loads(msg.content)
                                    tool_results.append(result)
                                except json.JSONDecodeError:
                                    tool_results.append({
                                        "status": "success",
                                        "content": msg.content,
                                        "source": "tool_response"
                                    })
                        except Exception as e:
                            print(f"Error parsing tool result: {e}")
            
            # Build context for response
            context_parts = [f"Current User Query: {state['user_query']}"]
            
            if conversation_summary:
                context_parts.append(f"\nConversation Summary: {conversation_summary}")
            
            if conversation_history:
                context_parts.append(f"\nRecent Conversation History:\n{conversation_history}")
            
            if tool_results:
                context_parts.append(f"\n--- External Information Retrieved ---")
                for i, result in enumerate(tool_results):
                    source = result.get('source', 'unknown_tool')
                    if result.get('status') == 'success' and result.get('context'):
                        context_parts.append(f"From {source}:\n{result['context']}")
            else:
                context_parts.append(f"\nNo external tools were used - responding based on conversation context.")
            
            full_context = "\n".join(context_parts)
            
            # Generate response
            system_message = """You are a helpful AI assistant with access to conversation memory. 

Key guidelines:
1. PRIORITIZE conversation context over general knowledge
2. Remember and use information the user has shared (like their name, preferences, etc.)
3. Reference previous parts of the conversation naturally
4. If the user asks about something they told you before, answer from conversation memory
5. Be conversational and build on previous exchanges
6. Only use external information when the conversation context isn't sufficient

Examples of good memory usage:
- User: "My name is John" → Remember this
- User: "What is my name?" → Answer: "Your name is John, as you told me earlier"
- User: "What did we discuss?" → Summarize previous conversation topics"""

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"""Based on our conversation and the context below, provide a helpful response:

{full_context}

Remember to:
- Use information from our conversation history
- Be natural and conversational
- Reference previous exchanges when relevant
- Answer directly if the information is in our conversation""")
            ]
            
            response = await self.llm.ainvoke(messages)
            final_response = response.content if hasattr(response, 'content') else str(response)
            
            state["final_response"] = final_response
            state["tool_results"] = tool_results
            
            return state
            
        except Exception as e:
            logger.error(f"Final responder error: {e}")
            state["final_response"] = f"I apologize, but I encountered an error: {str(e)}"
            return state


class LinearOrchestrator:
    """Enhanced orchestrator with better memory handling"""
    
    def __init__(self, st):
        self.st = st
        self.supervisor = SupervisorAgent(st)
        self.final_responder = FinalResponderAgent(st)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _should_skip_tools(self, state) -> bool:
        """Check if supervisor decided to skip tools"""
        return any(
            isinstance(msg, AIMessage) and msg.content == "NO_TOOLS_NEEDED"
            for msg in state.get("messages", [])
        )
    
    def _build_graph(self):
        """Build workflow with conditional tool usage"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self.supervisor.process)
        workflow.add_node("tools", tool_node)
        workflow.add_node("final_responder", self.final_responder.process)
        
        # Start with supervisor
        workflow.add_edge(START, "supervisor")
        
        # Conditional logic: supervisor -> tools (if needed) -> final_responder
        def should_use_tools(state):
            return "tools" if not self._should_skip_tools(state) else "final_responder"
        
        workflow.add_conditional_edges(
            "supervisor",
            should_use_tools,
            {
                "tools": "tools",
                "final_responder": "final_responder"
            }
        )
        
        workflow.add_edge("tools", "final_responder")
        workflow.add_edge("final_responder", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _get_session_id(self) -> str:
        """Get or create session ID for memory"""
        session_state = getattr(self.st, 'session_state', {})
        if 'session_id' not in session_state:
            session_state['session_id'] = str(uuid.uuid4())
        return session_state['session_id']
    
    async def process_query(self, user_query: str, available_files: List[Dict] = None) -> Dict[str, Any]:
        """Process query with enhanced memory handling"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            session_id = self._get_session_id()
            thread_config = {"configurable": {"thread_id": session_id}}
            
            # Get conversation history
            try:
                current_state = await self.graph.aget_state(thread_config)
                if current_state and hasattr(current_state, 'values') and current_state.values:
                    conversation_history = current_state.values.get('messages', [])
                    conversation_summary = current_state.values.get('conversation_summary')
                else:
                    conversation_history = []
                    conversation_summary = None
            except Exception as e:
                print(f"No previous conversation state: {e}")
                conversation_history = []
                conversation_summary = None
            
            # Create initial state
            initial_state = AgentState(
                messages=conversation_history,
                user_query=user_query,
                available_files=available_files or [],
                tool_results=[],
                final_response="",
                conversation_summary=conversation_summary,
                session_id=session_id
            )
            
            # Add current user query
            initial_state["messages"].append(HumanMessage(content=user_query))
            
            # Process with memory
            final_state = await self.graph.ainvoke(initial_state, config=thread_config)
            
            # Add final response to history
            if final_state["final_response"]:
                final_state["messages"].append(AIMessage(content=final_state["final_response"]))
            
            return {
                'status': 'success',
                'final_response': final_state["final_response"],
                'tool_results': final_state["tool_results"],
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                'status': 'error',
                'final_response': f"I encountered an error: {str(e)}",
                'error': str(e),
                'session_id': getattr(self, '_session_id', 'unknown')
            }
    
    async def clear_conversation_memory(self):
        """Clear conversation memory"""
        try:
            session_state = getattr(self.st, 'session_state', {})
            session_state['session_id'] = str(uuid.uuid4())
            logger.info("Conversation memory cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return False
    
    async def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        try:
            session_id = self._get_session_id()
            thread_config = {"configurable": {"thread_id": session_id}}
            
            current_state = await self.graph.aget_state(thread_config)
            if not current_state or not hasattr(current_state, 'values') or not current_state.values:
                return []
                
            messages = current_state.values.get('messages', [])
            
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content != "NO_TOOLS_NEEDED":
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []


# Test the enhanced memory system
async def main():
    """Test enhanced conversation memory"""
    class MockST:
        def __init__(self):
            self.session_state = {}
    
    st = MockST()
    orchestrator = LinearOrchestrator(st)
    
    print("=== Enhanced Conversation Memory Test ===")
    
    # First query - introduce name
    result1 = await orchestrator.process_query("My name is Aakash Khamaru and I'm interested in AI technology.")
    print("Query 1: My name is Aakash Khamaru and I'm interested in AI technology.")
    print(f"Response: {result1['final_response'][:200]}...")
    
    # Second query - ask about name (should remember)
    result2 = await orchestrator.process_query("What is my name?")
    print(f"\nQuery 2: What is my name?")
    print(f"Response: {result2['final_response']}")
    
    # Third query - ask about interests
    result3 = await orchestrator.process_query("What did I tell you I was interested in?")
    print(f"\nQuery 3: What did I tell you I was interested in?")
    print(f"Response: {result3['final_response']}")
    
    # Show conversation history
    history = await orchestrator.get_conversation_history()
    print(f"\nConversation History ({len(history)} messages):")
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    # Clear and test
    await orchestrator.clear_conversation_memory()
    print(f"\nMemory cleared!")

if __name__ == "__main__":
    asyncio.run(main())