import asyncio
import os
import uuid
import json
import httpx
import requests
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# -----------------------------
# STATE
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_query: str
    available_files: List[Dict]
    tool_results: List[Dict]
    final_response: str
    conversation_summary: Optional[str]
    session_id: str

# -----------------------------
# TOOLS (unchanged)
# -----------------------------
@tool
async def search_documents(query: str, file_ids: List[str] = None) -> Dict[str, Any]:
    """Search through uploaded documents for relevant information"""
    print(f"🔍 Document search called with query: {query}")
    
    payload = {"query": query, "collection": "sandbox", "top_k": 10}
    url = 'https://3ce4781230c0.ngrok-free.app/api/v1/chat/vector-search'
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url, json=payload,
                headers={"accept": "application/json", "Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            
        chunks = data.get("top_chunks", [])
        print(f"📄 Found {len(chunks)} chunks")
        
        if not chunks:
            return {
                "status": "success",
                "results": [],
                "context": "No relevant documents found for your query.",
                "total_results": 0,
                "source": "search_documents"
            }
        
        context_parts = []
        results = []
        
        for i, chunk in enumerate(chunks[:5]):
            filename = chunk.get('file_name', 'Unknown File')
            text = chunk.get('text', '').strip()
            page = chunk.get('page_number', 'N/A')
            
            if text:
                context_parts.append(f"**Document {i+1}: {filename} (Page {page})**\n{text}")
                
                results.append({
                    "filename": filename,
                    "content": text,
                    "page": page,
                    "section": chunk.get('section_title', 'N/A'),
                    "doc_id": chunk.get('source_doc_id', 'N/A'),
                    "similarity": chunk.get('similarity_score', 0)
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        result = {
            "status": "success",
            "results": results,
            "context": context or "No relevant content found in documents.",
            "total_results": len(chunks),
            "query": query,
            "source": "search_documents"
        }
        
        print(f"✅ Document search success: {len(results)} results")
        return result
        
    except Exception as e:
        error_result = {"status": "error", "error": str(e), "source": "search_documents"}
        print(f"❌ Document search error: {e}")
        return error_result

@tool
async def web_search(query: str, num_results: int = 5, search_type: str = "search") -> Dict[str, Any]:
    """Search the web using Serper.dev API"""
    try:
        print(f"🌐 Web search for: {query}")
        api_key = os.getenv('SERPER_API_KEY')
        if not api_key:
            return {"status": "error", "error": "SERPER_API_KEY missing", "source": "web_search"}

        url = "https://google.serper.dev/search" if search_type == "search" else "https://google.serper.dev/news"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(num_results, 10)}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = []
        items = data.get("organic", []) if search_type == "search" else data.get("news", [])
        for r in items[:num_results]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
                "source": r.get("source", ""),
                "date": r.get("date", "")
            })
        
        context = "\n\n".join([f"**{r['title']}**\n{r['snippet']}" for r in results[:3] if r.get('snippet')])

        print(f"✅ Web search success: {len(results)} results")
        return {
            "status": "success",
            "results": results,
            "context": context or "No results found.",
            "total_results": len(results),
            "source": "web_search"
        }
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return {"status": "error", "error": str(e), "source": "web_search"}

available_tools = [search_documents, web_search]
tool_node = ToolNode(available_tools)

# -----------------------------
# SUPERVISOR (Fixed)
# -----------------------------
class SupervisorAgent:
    def __init__(self, st):
        self.st = st
        self.llm = self._get_supervisor_llm()
        self.llm_with_tools = self.llm.bind_tools(available_tools)
    
    def _get_supervisor_llm(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    async def process(self, state: AgentState) -> AgentState:
        print(f"🤖 Supervisor processing query: {state['user_query']}")
        
        sys_prompt = SystemMessage(content="""You are a supervisor deciding which tool to use.

TOOL SELECTION RULES:
- Use `web_search` for: current events, news, recent updates, "what's happening", "latest", "today"
- Use `search_documents` for: questions about uploaded files, PDFs, documents, reports
- If the question can be answered from conversation history alone, respond without tools

You must either call a tool or provide a direct response - never both.""")
        
        # Include conversation history for context
        messages = [sys_prompt] + state["messages"]
        
        response = await self.llm_with_tools.ainvoke(messages)
        print(f"📋 Supervisor response has {len(getattr(response, 'tool_calls', []))} tool calls")
        
        # Return updated state with supervisor's response
        return {**state, "messages": state["messages"] + [response]}

# -----------------------------
# FINAL RESPONDER (Fixed)
# -----------------------------
class FinalResponderAgent:
    def __init__(self, st):
        self.st = st
        self.llm = self._get_responder_llm()
        
    def _get_responder_llm(self):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def _extract_conversation_context(self, messages):
        """Extract clean conversation context without incomplete tool calls"""
        clean_messages = []
        tool_context = []
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Handle system messages
            if isinstance(msg, SystemMessage):
                # Skip system messages for conversation context
                i += 1
                continue
            
            # Handle human messages
            elif isinstance(msg, HumanMessage):
                clean_messages.append(msg)
                i += 1
            
            # Handle AI messages with tool calls
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Look for corresponding tool messages
                    tool_call_ids = {tc.get('id') for tc in msg.tool_calls if isinstance(tc, dict) and 'id' in tc}
                    if not tool_call_ids:
                        # Handle tool_calls that are objects instead of dicts
                        tool_call_ids = {getattr(tc, 'id', None) for tc in msg.tool_calls if hasattr(tc, 'id')}
                    
                    # Find tool messages that respond to these calls
                    j = i + 1
                    found_responses = set()
                    tool_messages = []
                    
                    while j < len(messages) and isinstance(messages[j], ToolMessage):
                        tool_msg = messages[j]
                        if hasattr(tool_msg, 'tool_call_id') and tool_msg.tool_call_id in tool_call_ids:
                            found_responses.add(tool_msg.tool_call_id)
                            tool_messages.append(tool_msg)
                        j += 1
                    
                    # Only include this sequence if all tool calls have responses
                    if found_responses == tool_call_ids:
                        # Extract context from tool results instead of including raw messages
                        for tool_msg in tool_messages:
                            try:
                                content = tool_msg.content
                                if isinstance(content, str):
                                    content = json.loads(content)
                                
                                if content.get("status") == "success" and content.get("context"):
                                    source = content.get('source', 'tool')
                                    tool_context.append(f"Previous {source} result: {content['context'][:500]}...")
                            except:
                                continue
                        
                        i = j  # Skip past all the tool messages
                    else:
                        # Skip incomplete tool call sequence
                        i += 1
                else:
                    # Regular AI message without tool calls
                    clean_messages.append(msg)
                    i += 1
            
            # Handle standalone tool messages (shouldn't happen but just in case)
            elif isinstance(msg, ToolMessage):
                i += 1
                continue
            
            else:
                clean_messages.append(msg)
                i += 1
        
        return clean_messages, tool_context
    
    async def process(self, state: AgentState) -> AgentState:
        print("💬 Final responder generating response")
        
        # Get clean conversation context
        clean_messages, tool_context = self._extract_conversation_context(state["messages"])
        
        sys_prompt = SystemMessage(content="""You are a helpful AI assistant. 

Guidelines:
- Use conversation history to maintain context across turns
- When external information is provided, incorporate it naturally into your response
- Be conversational and remember what the user has asked before
- Provide helpful and informative responses""")

        # Build context information
        context_info = ""
        if tool_context:
            context_info = f"\n\nAdditional context from tools:\n" + "\n".join(tool_context)

        # Create the final prompt
        user_prompt = f"""Current question: {state['user_query']}{context_info}

Please provide a helpful response based on the conversation history and any available context."""

        # Use clean messages for the conversation
        messages = [sys_prompt] + clean_messages + [HumanMessage(content=user_prompt)]
        
        response = await self.llm.ainvoke(messages)
        print(f"✅ Generated response: {response.content[:100]}...")
        
        return {
            **state, 
            "final_response": response.content,
            "messages": state["messages"] + [response]
        }

# -----------------------------
# ORCHESTRATOR (unchanged)
# -----------------------------
class LinearOrchestrator:
    def __init__(self, st):
        self.st = st
        self.supervisor = SupervisorAgent(self.st)
        self.final_responder = FinalResponderAgent(self.st)
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
        if not hasattr(self.st, "session_state"):
            self.st.session_state = {}
        if "conversation_messages" not in self.st.session_state:
            self.st.session_state["conversation_messages"] = []
        if "session_id" not in self.st.session_state:
            self.st.session_state["session_id"] = str(uuid.uuid4())

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("supervisor", self.supervisor.process)
        g.add_node("tools", tool_node)
        g.add_node("responder", self.final_responder.process)

        g.add_edge(START, "supervisor")

        def route_after_supervisor(state):
            """Route based on whether supervisor made tool calls"""
            last_message = state["messages"][-1] if state["messages"] else None
            
            # Check if the last message (from supervisor) has tool calls
            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"🔧 Routing to tools - found {len(last_message.tool_calls)} tool calls")
                return "tools"
            else:
                print("💭 Routing to responder - no tool calls")
                return "responder"

        g.add_conditional_edges(
            "supervisor", 
            route_after_supervisor, 
            {"tools": "tools", "responder": "responder"}
        )
        g.add_edge("tools", "responder")
        g.add_edge("responder", END)
        
        return g.compile(checkpointer=self.memory)

    async def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n🚀 Processing query: {query}")
        
        session_id = self.st.session_state["session_id"]
        
        # Get conversation history and add new user message
        conversation_messages = self.st.session_state["conversation_messages"].copy()
        conversation_messages.append(HumanMessage(content=query))

        state = AgentState(
            messages=conversation_messages,
            user_query=query,
            available_files=[],
            tool_results=[],
            final_response="",
            conversation_summary=None,
            session_id=session_id
        )

        # Run the graph
        result = await self.graph.ainvoke(
            state, 
            config={"configurable": {"thread_id": session_id}}
        )

        # Update conversation history
        self.st.session_state["conversation_messages"] = result["messages"]

        print(f"✅ Final response ready: {result['final_response'][:100]}...")
        return {
            "response": result["final_response"], 
            "session": session_id
        }

# -----------------------------
# TESTING
# -----------------------------
async def test_memory_basic():
    class DummyStreamlit:
        def __init__(self):
            self.session_state = {}
        
    st = DummyStreamlit()
    orchestrator = LinearOrchestrator(st)

    # Turn 1
    print("\n" + "="*50)
    result1 = await orchestrator.process_query("Hi, my name is Alice. Who are you?")
    print("Turn 1 Response:", result1["response"])

    # Turn 2 - should remember the first turn
    print("\n" + "="*50)
    result2 = await orchestrator.process_query("What's my name?")
    print("Turn 2 Response:", result2["response"])

    # Turn 3 - test tool usage with memory
    print("\n" + "="*50)
    result3 = await orchestrator.process_query("Can you search for recent AI news?")
    print("Turn 3 Response:", result3["response"])

    # Turn 4 - reference previous searches
    print("\n" + "="*50)
    result4 = await orchestrator.process_query("What did you just search for?")
    print("Turn 4 Response:", result4["response"])

if __name__ == "__main__":
    asyncio.run(test_memory_basic())