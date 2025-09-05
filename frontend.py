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

# Import our agents and components
from agents import LinearOrchestrator
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

orchestrator = LinearOrchestrator(st)

class FileProcessor:
    def __init__(self):
        self.api_base_url = "https://3ce4781230c0.ngrok-free.app/api/v1/documents"
        self.collection_id = "sandbox"  # You can make this configurable
    
    def process_uploaded_file(self, uploaded_file):
        """Upload file to API and track processing status"""
        try:
            # Step 1: Upload file to API
            job_id = self._upload_file_to_api(uploaded_file)
            
            if not job_id:
                st.error(f"Failed to upload {uploaded_file.name}")
                return
            
            # Step 2: Track job status with Streamlit components
            self._track_job_status(job_id, uploaded_file.name)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    def _upload_file_to_api(self, uploaded_file) -> Optional[str]:
        """Upload file to the API endpoint"""
        try:
            # Prepare file for upload
            files = {
                'file': (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or 'application/octet-stream'
                )
            }
            
            # API endpoint
            upload_url = f"{self.api_base_url}/upload"
            params = {'collection': self.collection_id}
            headers = {'accept': 'application/json'}
            
            # Show upload progress
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


    def _track_job_status(self, job_id: str, poll_interval: int = 2, max_attempts: int = 30):
        """
        Poll API until job finishes or fails.
        Shows Streamlit progress updates.
        """
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
                            st.info(f"⏳ {job_id} is still {status}... (attempt {attempt+1}/{max_attempts})")
                        time.sleep(poll_interval)
                        continue

                    elif status in ["completed", "success", "done", "finished"]:
                        st.success(f"✅ {job_id} processing completed!")
                        return result

                    elif status in ["failed", "error"]:
                        st.error(f"❌ {job_id} processing failed: {result}")
                        return result

                    else:
                        st.warning(f"⚠️ {job_id} returned unknown status: {status}")
                        return result
                else:
                    st.error(f"Error fetching job status: {response.status_code} - {response.text}")
                    return None

            except requests.exceptions.RequestException as e:
                st.error(f"Status check error for job {job_id}: {e}")
                return None

        st.warning(f"⚠️ Gave up tracking job {job_id} after {max_attempts} attempts.")
        return None

class StreamlitUI:
    def __init__(self):
        self.db = db_handler
        self.prompts = prompts_manager
        self.file_processor = FileProcessor()
        
        # Initialize session state
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        # Initialize model preferences with defaults
        if 'brain_provider' not in st.session_state:
            st.session_state.brain_provider = "openai"
        if 'heart_provider' not in st.session_state:
            st.session_state.heart_provider = "anthropic"
    
    def run(self):
        """Main UI entry point"""
        st.title("🧠❤️ Brain-Heart AI System")
        st.markdown("""
        **Intelligent Query Processing with Parallel Subtask Execution**
        
        Upload files, ask complex questions, and watch the Brain-Heart system decompose, 
        process, and synthesize responses using advanced RAG and reasoning techniques.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_main_interface()
        with col2:
            self.render_sidebar()
        
        if st.session_state.last_result:
            st.divider()
            self.render_results_section()
    
    # -------------------- Main Interface --------------------
    def render_main_interface(self):
        st.header("💭 Query Interface")
        self.render_file_upload_section()
        
        st.subheader("Ask Your Question")
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="Ask me anything about your uploaded files or any topic...",
            help="Enter a complex question and watch the Brain-Heart system break it down into subtasks"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧠 Process with Brain-Heart System", type="primary", use_container_width=True):
                if query.strip():
                    self.process_query(query)
                else:
                    st.error("Please enter a query")
        
        if st.session_state.processing:
            self.render_processing_interface()
    
    # -------------------- File Upload --------------------
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
                        # ✅ Delegate file handling to FileProcessor
                        self.file_processor.process_uploaded_file(uploaded_file)
                        
                        # Optionally track uploaded file locally
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
                    # Just remove from session state (since API owns persistence now)
                    if st.button("🗑️", key=f"delete_{file_info['filename']}", help="Remove file"):
                        st.session_state.uploaded_files = [
                            f for f in st.session_state.uploaded_files 
                            if f['filename'] != file_info['filename']
                        ]
                        st.rerun()
    
    # -------------------- Sidebar --------------------
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
            
            # Show current model selection
            st.markdown("**Current Model Selection:**")
            st.text(f"Brain Agent: {st.session_state.brain_provider}")
            st.text(f"Heart Agent: {st.session_state.heart_provider}")
            
            llm_status = {}
            try:
                from tools import tool_executor
                for provider, llm in tool_executor.llms.items():
                    llm_status[provider] = "✅ Connected" if llm else "❌ Unavailable"
            except:
                llm_status = {"Status": "❌ Error"}
            
            for provider, status in llm_status.items():
                st.text(f"{provider.title()}: {status}")
    
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
            max_retries = st.slider("Max Retries for Critical Tasks", 1, 5, 3)
            critical_threshold = st.slider("Critical Weight Threshold", 0.5, 1.0, 0.8, 0.1)
            
            st.markdown("**LLM Provider Preferences:**")
            
            # Use callback to update session state when selection changes
            brain_provider = st.selectbox(
                "Brain Agent LLM", 
                ["openai", "anthropic", "openrouter"], 
                index=["openai", "anthropic", "openrouter"].index(st.session_state.brain_provider),
                key="brain_provider_selector"
            )
            
            heart_provider = st.selectbox(
                "Heart Agent LLM", 
                ["anthropic", "openai", "openrouter"], 
                index=["anthropic", "openai", "openrouter"].index(st.session_state.heart_provider),
                key="heart_provider_selector"
            )
            
            # Update session state immediately when selection changes
            if brain_provider != st.session_state.brain_provider:
                st.session_state.brain_provider = brain_provider
                st.success(f"Brain Agent LLM updated to: {brain_provider}")
                
            if heart_provider != st.session_state.heart_provider:
                st.session_state.heart_provider = heart_provider
                st.success(f"Heart Agent LLM updated to: {heart_provider}")
            
            # Store settings in session state for persistence
            st.session_state.max_retries = max_retries
            st.session_state.critical_threshold = critical_threshold
            
            if st.button("🗑️ Clear Cache"):
                from utils import cache
                cache.cache.clear()
                st.success("Cache cleared!")

            if st.button("🗑️ Delete Collection", type="primary"):
                try:
                    delete_url = f"https://3ce4781230c0.ngrok-free.app/api/v1/collections/"
                    params = {"collection_name": self.file_processor.collection_id}
                    headers = {"accept": "application/json"}
                    resp = requests.delete(delete_url, params=params, headers=headers, timeout=30)

                    if resp.status_code == 200:
                        st.success(f"✅ Collection '{self.file_processor.collection_id}' deleted successfully!")
                    else:
                        st.error(f"❌ Failed to delete collection: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error deleting collection: {e}")
    
    # -------------------- Processing --------------------
    def render_processing_interface(self):
        st.subheader("🔄 Processing Status")
        progress_container = st.container()
        
        with progress_container:
            progress_data = progress_tracker.get_progress()
            overall_progress = progress_data['progress_percentage'] / 100
            st.progress(overall_progress, text=f"Overall Progress: {progress_data['progress_percentage']:.1f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", progress_data['total_tasks'])
            with col2:
                st.metric("Completed", progress_data['completed_tasks'])
            with col3:
                st.metric("Running", progress_data['running_tasks'])
            with col4:
                st.metric("Failed", progress_data['failed_tasks'])
            
            if progress_data['tasks']:
                st.markdown("**Task Details:**")
                for task_id, task_info in progress_data['tasks'].items():
                    status_emoji = {'pending': '⏳', 'running': '🔄', 'completed': '✅', 'failed': '❌'}
                    emoji = status_emoji.get(task_info['status'], '❓')
                    weight_bar = "█" * int(task_info['weight'] * 10)
                    st.text(f"{emoji} {task_id}: {task_info['description']}")
                    st.text(f"   Weight: {weight_bar} ({task_info['weight']:.1f})")
                    if task_info['status'] == 'failed' and task_info.get('error'):
                        st.error(f"   Error: {task_info['error']}")
    
    # -------------------- Results --------------------
    def render_results_section(self):
        st.header("📊 Results")
        result = st.session_state.last_result
        if result['status'] == 'error':
            st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            return
        
        st.subheader("💬 Final Response")
        st.markdown(result['final_response'])
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "🧠 Brain Results", "❤️ Heart Results", "🔍 Raw Data"])
        with tab1: self.render_analysis_tab(result)
        with tab2: self.render_brain_results_tab(result.get('brain_result', {}))
        with tab3: self.render_heart_results_tab(result.get('heart_result', {}))
        with tab4: self.render_raw_data_tab(result)
    
    def render_analysis_tab(self, result: Dict):
        st.subheader("Processing Analysis")
        brain_result = result.get('brain_result', {})
        heart_result = result.get('heart_result', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "green" if result['status'] == 'success' else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}'>{result['status']}</span>", unsafe_allow_html=True)
        with col2:
            fallback_used = result.get('used_fallback', False)
            fallback_color = "orange" if fallback_used else "green"
            fallback_text = "Yes" if fallback_used else "No"
            st.markdown(f"**Fallback Used:** <span style='color:{fallback_color}'>{fallback_text}</span>", unsafe_allow_html=True)
        with col3:
            processing_time = brain_result.get('processing_time', 0)
            st.markdown(f"**Processing Time:** {processing_time:.2f}s")
        
        if brain_result.get('results'):
            self.render_task_execution_chart(brain_result['results'])
        if heart_result.get('qa_assessment'):
            self.render_quality_assessment(heart_result['qa_assessment'])
    
    def render_task_execution_chart(self, task_results: List[Dict]):
        st.subheader("Task Execution Overview")
        tasks_data = []
        for result in task_results:
            tasks_data.append({
                'Task ID': result.get('task_id', 'Unknown'),
                'Status': result.get('status', 'Unknown'),
                'Weight': result.get('weight', 0.5),
                'Type': result.get('task_type', 'Unknown'),
                'LLM Provider': result.get('llm_provider', 'Unknown')
            })
        df = pd.DataFrame(tasks_data)
        
        col1, col2 = st.columns(2)
        with col1:
            status_counts = df['Status'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3,
                marker_colors=['#2E8B57', '#DC143C', '#FFA500', '#4169E1']
            )])
            fig_pie.update_layout(title="Task Status Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            color_map = {'completed': '#2E8B57', 'failed': '#DC143C', 'pending': '#FFA500'}
            fig_scatter = go.Figure()
            for status in df['Status'].unique():
                status_data = df[df['Status'] == status]
                fig_scatter.add_trace(go.Scatter(
                    x=status_data.index,
                    y=status_data['Weight'],
                    mode='markers',
                    name=status,
                    marker=dict(size=15, color=color_map.get(status, '#808080'), opacity=0.7),
                    text=status_data['Task ID'],
                    hovertemplate='<b>%{text}</b><br>Weight: %{y}<br>Status: ' + status
                ))
            fig_scatter.update_layout(title="Task Weight vs Execution Order", xaxis_title="Execution Order", yaxis_title="Task Weight", height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Task Details")
        st.dataframe(df, use_container_width=True)
    
    def render_quality_assessment(self, qa_assessment: Dict):
        st.subheader("Response Quality Assessment")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            content_color = "green" if qa_assessment.get('has_content') else "red"
            st.markdown(f"**Has Content:** <span style='color:{content_color}'>{'✅' if qa_assessment.get('has_content') else '❌'}</span>", unsafe_allow_html=True)
        with col2:
            rag_color = "green" if qa_assessment.get('has_rag_context') else "orange"
            st.markdown(f"**RAG Context:** <span style='color:{rag_color}'>{'✅' if qa_assessment.get('has_rag_context') else '⚠️'}</span>", unsafe_allow_html=True)
        with col3:
            analysis_color = "green" if qa_assessment.get('has_analysis') else "orange"
            st.markdown(f"**Analysis:** <span style='color:{analysis_color}'>{'✅' if qa_assessment.get('has_analysis') else '⚠️'}</span>", unsafe_allow_html=True)
        with col4:
            confidence_score = qa_assessment.get('confidence_score', 0)
            confidence_color = "green" if confidence_score > 0.7 else "orange" if confidence_score > 0.4 else "red"
            st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence_score:.2f}</span>", unsafe_allow_html=True)
    
    # -------------------- Brain / Heart / Raw --------------------
    def render_brain_results_tab(self, brain_result: Dict):
        st.subheader("🧠 Brain Agent Output")
        if not brain_result:
            st.info("No Brain agent data available")
            return
        
        subtasks = brain_result.get('subtasks', [])
        if subtasks:
            st.markdown("**Generated Subtasks:**")
            for i, task in enumerate(subtasks, 1):
                with st.expander(f"Task {i}: {task.get('task_id', 'Unknown')}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            'Description': task.get('description', ''),
                            'Weight': task.get('weight', 0),
                            'Dependencies': task.get('depends_on', []),
                            'Type': task.get('task_type', ''),
                            'LLM Provider': task.get('llm_provider', '')
                        })
                    with col2:
                        st.markdown("**Parameters:**")
                        st.json(task.get('parameters', {}))
        
        results = brain_result.get('results', [])
        if results:
            st.markdown("**Execution Results:**")
            for result in results:
                status_emoji = "✅" if result.get('status') == 'completed' else "❌"
                with st.expander(f"{status_emoji} {result.get('task_id', 'Unknown')}", expanded=False):
                    if result.get('status') == 'completed':
                        st.success("Task completed successfully")
                        st.markdown("**Result:**")
                        st.text_area("", value=str(result.get('result', '')), height=100, disabled=True)
                        if result.get('context'):
                            st.markdown("**Retrieved Context:**")
                            for ctx in result['context'][:3]:
                                st.markdown(f"- Similarity: {ctx.get('similarity', 0):.3f}")
                                st.text(ctx.get('text', '')[:200] + "..." if len(ctx.get('text', '')) > 200 else ctx.get('text', ''))
                    else:
                        st.error(f"Task failed: {result.get('error', 'Unknown error')}")
    
    def render_heart_results_tab(self, heart_result: Dict):
        st.subheader("❤️ Heart Agent Output")
        if not heart_result:
            st.info("No Heart agent data available")
            return
        
        analysis = heart_result.get('analysis', {})
        if analysis:
            st.markdown("**Response Assembly Analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Success Rate", f"{analysis.get('success_rate', 0):.2%}")
                st.metric("Completed Tasks", f"{analysis.get('completed_tasks', 0)}/{analysis.get('total_tasks', 0)}")
            with col2:
                critical_failures = analysis.get('has_critical_failures', False)
                st.markdown(f"**Critical Failures:** {'❌ Yes' if critical_failures else '✅ No'}")
                st.metric("Failed Tasks", analysis.get('failed_tasks', 0))
        
        qa_assessment = heart_result.get('qa_assessment', {})
        if qa_assessment:
            with st.expander("Quality Assessment Details", expanded=True):
                st.json(qa_assessment)
        
        if heart_result.get('used_fallback'):
            st.warning("⚠️ Fallback strategies were used due to incomplete subtask results")
    
    def render_raw_data_tab(self, result: Dict):
        st.subheader("🔍 Raw Processing Data")
        st.json(result)
    
    # -------------------- File Handling --------------------
    def process_uploaded_file(self, uploaded_file):
        try:
            content = uploaded_file.read()
            file_type = uploaded_file.type or "unknown"
            file_id = self.db.save_file(
                filename=uploaded_file.name,
                content=content,
                file_type=file_type,
                metadata={'size': len(content)}
            )
            
            if file_type.startswith('text/') or uploaded_file.name.endswith(('.txt', '.md', '.py', '.js', '.html', '.css')):
                text_content = content.decode('utf-8', errors='ignore')
                chunks = self.chunk_text(text_content)
                self.db.index_file_content(file_id, chunks)
            else:
                metadata_text = f"File: {uploaded_file.name}, Type: {file_type}, Size: {len(content)} bytes"
                self.db.index_file_content(file_id, [metadata_text])
            
            st.session_state.uploaded_files.append({
                'file_id': file_id,
                'filename': uploaded_file.name,
                'file_type': file_type,
                'size': len(content)
            })
            st.success(f"✅ {uploaded_file.name} uploaded and indexed successfully!")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]
    
    def remove_file(self, file_id: str):
        try:
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['file_id'] != file_id]
            st.success("File removed successfully!")
        except Exception as e:
            st.error(f"Error removing file: {e}")
    
    # -------------------- Query Execution --------------------
    def process_query(self, query: str):
        st.session_state.processing = True
        progress_placeholder = st.empty()
        
        try:
            with progress_placeholder.container():
                st.info("🧠 Brain-Heart system is processing your query...")
                
                # Show current model selection in processing
                st.info(
                    f"Using Brain Agent: {st.session_state.brain_provider} | "
                    f"Heart Agent: {st.session_state.heart_provider}"
                )
                
                available_files = self.db.get_all_files()
                
                # ✅ Create configuration dictionary to pass model preferences
                config = {
                    "brain_provider": st.session_state.get("brain_provider", "openai"),
                    "heart_provider": st.session_state.get("heart_provider", "anthropic"),
                    "max_retries": st.session_state.get("max_retries", 3),
                    "critical_threshold": st.session_state.get("critical_threshold", 0.8),
                }
                
                # ✅ Pass config to orchestrator (new 3rd argument)
                result = asyncio.run(orchestrator.process_query(query, available_files))

                # Save result + history
                st.session_state.last_result = result
                st.session_state.chat_history.append({
                    "query": query,
                    "response": result["final_response"],
                    "timestamp": time.time(),
                    "status": result["status"],
                })

            progress_placeholder.empty()
            st.success("✅ Processing completed!")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Processing failed: {e}")
            st.session_state.last_result = {
                "status": "error",
                "final_response": f"I apologize, but I encountered an error: {e}",
                "error": str(e),
            }
        finally:
            st.session_state.processing = False


def run_frontend():
    ui = StreamlitUI()
    ui.run()


if __name__ == "__main__":
    run_frontend()