import streamlit as st
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from streamlit_ace import st_ace
import requests
import io
from datetime import datetime, timedelta
import uuid

# Import our agents and components
from new_agents import LinearOrchestrator  # Updated import
from database import db_handler
from prompts import prompts_manager, DEFAULT_PROMPTS
from utils import progress_tracker

# Page configuration
st.set_page_config(
    page_title="🧠❤️ Brain-Heart AI System",
    page_icon="🧠❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize orchestrator with streamlit instance
orchestrator = LinearOrchestrator(st)  # Pass st instance


def create_collection(name: str):
    url = f"https://3ce4781230c0.ngrok-free.app/api/v1/collections"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "name": name
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    # Raise an error if request failed
    response.raise_for_status()
    
    return response.json()


class ConversationMemory:
    """Simplified conversation memory manager for display purposes"""
    
    def __init__(self, max_history: int = 10, context_window: int = 5):
        self.max_history = max_history
        self.context_window = context_window
    
    def get_conversation_summary(self, chat_history: List[Dict]) -> str:
        """Get a summary of the conversation for display"""
        if not chat_history:
            return "No conversation history"
        
        # Extract topics from recent messages
        topics = []
        for msg in chat_history[-5:]:  # Last 5 messages
            query_words = msg.get('query', '').lower().split()
            topics.extend([w for w in query_words if len(w) > 4 and w.isalpha()])
        
        unique_topics = list(set(topics))[:3]  # Top 3 unique topics
        
        return f"{len(chat_history)} messages | Recent topics: {', '.join(unique_topics)}"


class FileProcessor:
    def __init__(self):
        self.api_base_url = "https://3ce4781230c0.ngrok-free.app/api/v1/documents"
        self.collection_id = "sandbox"
    
    def process_uploaded_file(self, uploaded_file):
        """Upload file to API and track processing status"""
        try:
            job_id = self._upload_file_to_api(uploaded_file)
            
            if not job_id:
                st.error(f"Failed to upload {uploaded_file.name}")
                return
            
            self._track_job_status(job_id, uploaded_file.name)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    def _upload_file_to_api(self, uploaded_file) -> Optional[str]:
        """Upload file to the API endpoint"""
        try:
            files = {
                'file': (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or 'application/octet-stream'
                )
            }
            
            upload_url = f"{self.api_base_url}/upload"
            params = {'collection': self.collection_id}
            headers = {'accept': 'application/json'}
            
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                response = requests.post(
                    upload_url,
                    params=params,
                    files=files,
                    headers=headers,
                    timeout=60
                )
            
            if response.status_code == 202:
                result = response.json()
                job_id = result.get('job_id') or result.get('id') or result.get('task_id')
                
                if job_id:
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    st.info(f"Job ID: `{job_id}`")
                    return job_id
                else:
                    st.error(f"Upload successful but no job ID returned: {result}")
                    return None
            else:
                st.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error(f"Upload timeout for {uploaded_file.name}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Upload error for {uploaded_file.name}: {e}")
            return None

    def _track_job_status(self, job_id: str, filename: str, poll_interval: int = 2, max_attempts: int = 30):
        """Poll API until job finishes or fails"""
        status_url = f"{self.api_base_url}/status/{job_id}"
        progress_placeholder = st.empty()

        for attempt in range(max_attempts):
            try:
                response = requests.get(status_url, timeout=20)
                if response.status_code == 200:
                    result = response.json()
                    status = str(result.get("status", "")).lower()

                    if status in ["queued", "queued_for_processing", "processing", "in_progress"]:
                        with progress_placeholder.container():
                            st.info(f"⏳ {filename} is still {status}... (attempt {attempt+1}/{max_attempts})")
                        time.sleep(poll_interval)
                        continue

                    elif status in ["completed", "success", "done", "finished"]:
                        progress_placeholder.success(f"✅ {filename} processing completed!")
                        return result

                    elif status in ["failed", "error"]:
                        progress_placeholder.error(f"❌ {filename} processing failed: {result}")
                        return result

                    else:
                        progress_placeholder.warning(f"⚠️ {filename} returned unknown status: {status}")
                        return result
                else:
                    st.error(f"Error fetching job status: {response.status_code} - {response.text}")
                    return None

            except requests.exceptions.RequestException as e:
                st.error(f"Status check error for job {job_id}: {e}")
                return None

        st.warning(f"⚠️ Gave up tracking {filename} after {max_attempts} attempts.")
        return None


class StreamlitUI:
    def __init__(self):
        self.db = db_handler
        self.prompts = prompts_manager
        self.file_processor = FileProcessor()
        self.memory = ConversationMemory()
        
        # Initialize session state
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = f"conv_{int(time.time())}"
        # UPDATED: Session management for new orchestrator
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        # UPDATED: Simplified provider settings for new orchestrator
        if 'supervisor_provider' not in st.session_state:
            st.session_state.supervisor_provider = "openai"
        if 'responder_provider' not in st.session_state:
            st.session_state.responder_provider = "openai"
    
    def run(self):
        """Main UI entry point"""
        st.title("🧠❤️ Enhanced AI System with Conversation Memory")
        st.markdown("""
        **Intelligent Query Processing with Built-in Memory**
        
        Upload files, ask questions, and engage in contextual conversations. 
        The system automatically remembers your conversation history.
        """)
        
        # UPDATED: Show memory status
        self._show_memory_status()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_main_interface()
        with col2:
            self.render_sidebar()
        
        # Conversation History Tab
        if st.session_state.chat_history:
            st.divider()
            self.render_conversation_history()
        
        if st.session_state.last_result:
            st.divider()
            self.render_results_section()
    
    def _show_memory_status(self):
        """Show current memory status"""
        if st.session_state.chat_history:
            summary = self.memory.get_conversation_summary(st.session_state.chat_history)
            st.info(f"💭 Conversation memory: {summary}")
            st.info(f"🔗 Current Session: {st.session_state.current_session_id[:8]}...")
    
    def render_main_interface(self):
        st.header("💭 Query Interface")
        self.render_file_upload_section()
        
        st.subheader("Ask Your Question")
        
        # UPDATED: Simplified query interface
        placeholder_text = "Ask me anything about your uploaded files or any topic..."
        if st.session_state.chat_history:
            placeholder_text = "Continue our conversation..."
        
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder=placeholder_text,
            help="The system automatically uses conversation history for context"
        )
        
        # UPDATED: Simplified controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Show session info
            st.caption(f"Session: {st.session_state.current_session_id[:8]}...")
        
        with col2:
            if st.button("🧹 Clear Memory", help="Clear conversation history"):
                # UPDATED: Clear memory by starting new session
                self._clear_conversation_memory()
                st.success("Conversation memory cleared!")
                time.sleep(1)
                st.rerun()
        
        with col3:
            if st.button("🧠 Process Query", type="primary", use_container_width=True):
                if query.strip():
                    self.process_query(query)
                else:
                    st.error("Please enter a query")
        
        if st.session_state.processing:
            self.render_processing_interface()
    
    def _clear_conversation_memory(self):
        """Clear conversation memory by creating new session"""
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.conversation_id = f"conv_{int(time.time())}"
    
    async def _get_conversation_history_async(self):
        """Get conversation history from LangGraph memory"""
        try:
            # UPDATED: Use the memory system from the orchestrator
            from langgraph.checkpoint.base import BaseCheckpointSaver
            
            # Get checkpoints for current session
            config = {"configurable": {"thread_id": st.session_state.current_session_id}}
            
            # Try to get history from the memory saver
            history = []
            try:
                # This is a simplified approach - you might need to adjust based on your actual memory implementation
                checkpoint = orchestrator.memory.get(config)
                if checkpoint:
                    messages = checkpoint.get('channel_values', {}).get('messages', [])
                    for msg in messages:
                        if hasattr(msg, 'content'):
                            history.append({
                                'role': getattr(msg, 'type', 'unknown'),
                                'content': msg.content
                            })
            except Exception as e:
                st.error(f"Error accessing memory: {e}")
                
            return history
        except Exception as e:
            st.error(f"Error getting conversation history: {e}")
            return []
    
    def render_file_upload_section(self):
        st.subheader("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Upload files for analysis",
            accept_multiple_files=True,
            help="Upload any type of file - they will be indexed for RAG retrieval"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['filename'] for f in st.session_state.uploaded_files]:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        self.file_processor.process_uploaded_file(uploaded_file)
                        
                        st.session_state.uploaded_files.append({
                            "filename": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "size": uploaded_file.size if hasattr(uploaded_file, "size") else None
                        })
        
        if st.session_state.uploaded_files:
            st.markdown("**Uploaded Files:**")
            for file_info in st.session_state.uploaded_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📄 {file_info['filename']} ({file_info['file_type']})")
                with col2:
                    if st.button("🗑️", key=f"delete_{file_info['filename']}", help="Remove file"):
                        st.session_state.uploaded_files = [
                            f for f in st.session_state.uploaded_files 
                            if f['filename'] != file_info['filename']
                        ]
                        st.rerun()
    
    def render_conversation_history(self):
        st.header("💬 Conversation History")
        
        # UPDATED: History controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Session:** `{st.session_state.current_session_id[:12]}...`")
        with col2:
            if st.button("📥 Export Chat", help="Export conversation as JSON"):
                chat_data = {
                    "conversation_id": st.session_state.conversation_id,
                    "session_id": st.session_state.current_session_id,
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.chat_history
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"conversation_{st.session_state.conversation_id}.json",
                    mime="application/json"
                )
        with col3:
            if st.button("📜 Show LangGraph History", help="Show internal conversation history"):
                # UPDATED: Use async function to get history
                try:
                    history = asyncio.run(self._get_conversation_history_async())
                    with st.expander("LangGraph Memory History", expanded=True):
                        if history:
                            for i, msg in enumerate(history):
                                st.text(f"{i+1}. {msg['role']}: {msg['content'][:200]}...")
                        else:
                            st.info("No LangGraph history available")
                except Exception as e:
                    st.error(f"Error fetching LangGraph history: {e}")
        
        # Display local chat history
        for i, msg in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            timestamp = datetime.fromtimestamp(msg.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S")
            status_emoji = "✅" if msg.get('status') == 'success' else "❌"
            
            with st.expander(f"{status_emoji} [{timestamp}] {msg.get('query', '')[:50]}...", expanded=False):
                st.markdown("**Query:**")
                st.text(msg.get('query', ''))
                st.markdown("**Response:**")
                st.text(msg.get('response', ''))
                
                # UPDATED: Show memory usage info
                if msg.get('session_id'):
                    st.info(f"🔗 Session: {msg['session_id'][:12]}...")
    
    def render_sidebar(self):
        st.header("⚙️ System Controls")
        self.render_system_status()
        
        st.subheader("📝 Prompt Editor")
        self.render_prompt_editor()
        
        st.subheader("🔧 Settings")
        self.render_settings()
    
    def render_system_status(self):
        with st.expander("🔍 System Status", expanded=False):
            try:
                files_count = len(self.db.get_all_files())
                st.metric("Indexed Files", files_count)
            except:
                st.metric("Indexed Files", "Error")
            
            if self.db.faiss_index:
                st.metric("Embeddings", self.db.faiss_index.ntotal)
            else:
                st.metric("Embeddings", "Error")
            
            # UPDATED: Show new provider settings
            st.markdown("**Current Model Selection:**")
            st.text(f"Supervisor: {st.session_state.supervisor_provider}")
            st.text(f"Responder: {st.session_state.responder_provider}")
            
            # Memory status
            st.markdown("**Memory Status:**")
            st.text(f"Local Messages: {len(st.session_state.chat_history)}")
            st.text(f"Session ID: {st.session_state.current_session_id[:12]}...")
            
            # Show LangGraph memory status
            try:
                history = asyncio.run(self._get_conversation_history_async())
                st.text(f"LangGraph Messages: {len(history)}")
            except:
                st.text("LangGraph Messages: Error")
    
    def render_prompt_editor(self):
        with st.expander("Edit System Prompts", expanded=False):
            prompt_types = self.prompts.get_all_prompt_types()
            selected_prompt = st.selectbox("Select Prompt Type", prompt_types)
            
            if selected_prompt:
                current_prompt = self.db.get_prompt(selected_prompt) or DEFAULT_PROMPTS.get(selected_prompt, "")
                edited_prompt = st_ace(
                    value=current_prompt,
                    language='text',
                    theme='github',
                    key=f"prompt_editor_{selected_prompt}",
                    height=200,
                    wrap=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Prompt", key=f"save_{selected_prompt}"):
                        try:
                            self.prompts.save_prompt(selected_prompt, edited_prompt)
                            st.success("Prompt saved!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving prompt: {e}")
                with col2:
                    if st.button("🔄 Reset to Default", key=f"reset_{selected_prompt}"):
                        try:
                            self.prompts.reset_prompt_to_default(selected_prompt)
                            st.success("Prompt reset!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error resetting prompt: {e}")
    
    def render_settings(self):
        with st.expander("System Settings", expanded=False):
            st.markdown("**LLM Provider Settings:**")
            
            # UPDATED: New provider settings for the new orchestrator
            supervisor_provider = st.selectbox(
                "Supervisor Agent LLM", 
                ["openai", "openrouter"], 
                index=["openai", "openrouter"].index(st.session_state.supervisor_provider),
                key="supervisor_provider_selector"
            )
            
            responder_provider = st.selectbox(
                "Final Responder LLM", 
                ["openai", "openrouter"], 
                index=["openai", "openrouter"].index(st.session_state.responder_provider),
                key="responder_provider_selector"
            )
            
            if supervisor_provider != st.session_state.supervisor_provider:
                st.session_state.supervisor_provider = supervisor_provider
                st.success(f"Supervisor LLM updated to: {supervisor_provider}")
                
            if responder_provider != st.session_state.responder_provider:
                st.session_state.responder_provider = responder_provider
                st.success(f"Responder LLM updated to: {responder_provider}")
            
            # Memory settings
            st.markdown("**Memory Settings:**")
            if st.button("🔄 Reset LangGraph Memory", help="Clear LangGraph conversation memory"):
                try:
                    self._clear_conversation_memory()
                    st.success("✅ LangGraph memory cleared!")
                except Exception as e:
                    st.error(f"Error clearing memory: {e}")
            
            if st.button("🗑️ Clear Cache"):
                from utils import cache
                if hasattr(cache, 'cache'):
                    cache.cache.clear()
                    st.success("Cache cleared!")
                else:
                    st.info("No cache to clear")

            if st.button("🗑️ Delete Collection", type="secondary"):
                try:
                    delete_url = f"https://3ce4781230c0.ngrok-free.app/api/v1/collections/"
                    params = {"collection_name": self.file_processor.collection_id}
                    headers = {"accept": "application/json"}
                    resp = requests.delete(delete_url, params=params, headers=headers, timeout=30)

                    if resp.status_code == 200:
                        st.success(f"✅ Collection '{self.file_processor.collection_id}' deleted successfully!")
                        create_collection('sandbox')
                    else:
                        st.error(f"❌ Failed to delete collection: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error deleting collection: {e}")
    
    def render_processing_interface(self):
        st.subheader("🔄 Processing Status")
        
        # Simple progress indicator for the new system
        with st.container():
            st.info("🧠 Processing query with conversation memory...")
            progress_bar = st.progress(0)
            
            # Simulate progress (you could enhance this with actual progress tracking)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
    
    def render_results_section(self):
        st.header("📊 Results")
        result = st.session_state.last_result
        
        if result['status'] == 'error':
            st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            return
        
        st.subheader("💬 Final Response")
        st.markdown(result['final_response'])
        
        # UPDATED: Show memory and session info
        col1, col2, col3 = st.columns(3)
        with col1:
            if result.get('session_id'):
                st.info(f"🔗 Session: {result['session_id'][:12]}...")
        with col2:
            tool_count = len(result.get('tool_results', []))
            st.info(f"🔧 Tools used: {tool_count}")
        with col3:
            status_color = "green" if result['status'] == 'success' else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}'>{result['status']}</span>", unsafe_allow_html=True)
        
        # UPDATED: Simplified tabs for new system
        tab1, tab2 = st.tabs(["🔍 Tool Results", "🔍 Raw Data"])
        
        with tab1:
            self.render_tool_results_tab(result.get('tool_results', []))
        with tab2:
            self.render_raw_data_tab(result)
    
    def render_tool_results_tab(self, tool_results: List[Dict]):
        st.subheader("🔧 Tool Execution Results")
        
        if not tool_results:
            st.info("No tools were used for this query")
            return
        
        for i, result in enumerate(tool_results):
            source = result.get('source', 'unknown_tool')
            status = result.get('status', 'unknown')
            
            status_emoji = "✅" if status == 'success' else "❌"
            
            with st.expander(f"{status_emoji} {source.title()} - {status}", expanded=False):
                if status == 'success':
                    if result.get('context'):
                        st.markdown("**Retrieved Context:**")
                        st.text_area("Context", value=result['context'], height=200, disabled=True)
                    
                    if result.get('results'):
                        st.markdown(f"**Results:** {result.get('total_results', 0)} items found")
                        
                        # Show first few results
                        for j, item in enumerate(result['results'][:3]):
                            st.markdown(f"**Result {j+1}:**")
                            if isinstance(item, dict):
                                st.json(item)
                            else:
                                st.text(str(item))
                else:
                    st.error(f"Tool failed: {result.get('error', 'Unknown error')}")
    
    def render_raw_data_tab(self, result: Dict):
        st.subheader("🔍 Raw Processing Data")
        st.json(result)
    
    def process_query(self, query: str):
        """UPDATED: Process query using new LinearOrchestrator"""
        st.session_state.processing = True
        
        try:
            with st.spinner("🧠 Processing query..."):
                # UPDATED: Use the new orchestrator interface with session management
                result = asyncio.run(self._process_query_with_session(query))
                
                # UPDATED: Handle the new response format
                final_response = result.get("response", "No response received")
                session_id = result.get("session", st.session_state.current_session_id)
                
                # Store result and update history
                processed_result = {
                    "status": "success",
                    "final_response": final_response,
                    "session_id": session_id,
                    "tool_results": [],  # The new orchestrator doesn't return detailed tool results
                }
                
                st.session_state.last_result = processed_result
                st.session_state.chat_history.append({
                    "query": query,
                    "response": final_response,
                    "timestamp": time.time(),
                    "status": "success",
                    "session_id": session_id,
                    "conversation_id": st.session_state.conversation_id
                })
                
                # Update current session ID if it changed
                st.session_state.current_session_id = session_id

            st.success("✅ Processing completed!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.session_state.last_result = {
                "status": "error",
                "final_response": f"I apologize, but I encountered an error: {e}",
                "error": str(e),
                "session_id": st.session_state.current_session_id,
                "tool_results": []
            }
        finally:
            st.session_state.processing = False
    
    async def _process_query_with_session(self, query: str):
        """Process query with session management for memory continuity"""
        from agents import AgentState
        from langchain_core.messages import HumanMessage
        
        # UPDATED: Use existing session ID for continuity or current for new queries
        session_id = st.session_state.current_session_id
        
        # Create state for the orchestrator
        state = AgentState(
            messages=[HumanMessage(content=query)],
            user_query=query,
            available_files=st.session_state.uploaded_files,
            tool_results=[],
            final_response="",
            conversation_summary=None,
            session_id=session_id
        )
        
        # Use the graph directly with session management
        config = {"configurable": {"thread_id": session_id}}
        result = await orchestrator.graph.ainvoke(state, config=config)
        
        return {
            "response": result.get("final_response", "No response generated"),
            "session": session_id
        }


def run_frontend():
    ui = StreamlitUI()
    ui.run()


if __name__ == "__main__":
    run_frontend()