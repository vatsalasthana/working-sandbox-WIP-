# import asyncio
# from typing import Dict, List, Any, Optional
# import json
# import os
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain.schema import HumanMessage, SystemMessage
# from database import db_handler
# from prompts import prompts_manager
# from utils import async_retry, logger, cache
# import aiohttp
# import numpy as np
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# class ToolExecutor:
#     def __init__(self):
#         self.db = db_handler
#         self.prompts = prompts_manager
#         self.collection='sandbox'
#         self.api_base_url="https://3ce4781230c0.ngrok-free.app/"
        
#         # Initialize LLM providers
#         self._initialize_llms()
    
#     def _initialize_llms(self):
#         """Initialize LangChain LLM providers"""
#         self.llms = {}
        
#         try:
#             # OpenAI
#             # if os.getenv('OPENAI_API_KEY'):
#             #     self.llms['openai'] = ChatOpenAI(
#             #         model="gpt-4.1-nano",
#             #         temperature=0.7,
#             #         openai_api_key=os.getenv('OPENAI_API_KEY')
#             #     )
            
#             # # Anthropic
#             # if os.getenv('ANTHROPIC_API_KEY'):
#             #     self.llms['anthropic'] = ChatAnthropic(
#             #         model="claude-3-sonnet-20240229",
#             #         temperature=0.7,
#             #         anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
#             #     )
            
#             # OpenRouter (as fallback)
#             if os.getenv('OPENROUTER_API_KEY'):
#                 self.llms['openrouter'] = ChatOpenAI(
#                     model="deepseek/deepseek-chat-v3.1",
#                     temperature=0.7,
#                     openai_api_key=os.getenv('OPENROUTER_API_KEY'),
#                     openai_api_base="https://openrouter.ai/api/v1"
#                 )
            
#             logger.info(f"Initialized {len(self.llms)} LLM providers: {list(self.llms.keys())}")
            
#         except Exception as e:
#             logger.error(f"Error initializing LLMs: {e}")
    
#     def get_llm(self, provider: str):
#         """Get LLM instance by provider name"""
#         return self.llms.get(provider, self.llms.get('openai'))  # Default to OpenAI
    
#     @async_retry(max_retries=2)
#     async def run_tool(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute a subtask using the appropriate tool"""
#         task_type = subtask.get('task_type', 'analysis')
#         task_id = subtask.get('task_id', 'unknown')
        
#         try:
#             if task_type == 'rag':
#                 return await self.execute_rag_task(subtask)
#             elif task_type == 'rat':
#                 return await self.execute_rat_task(subtask)
#             elif task_type == 'web_search':
#                 return await self.execute_web_search_task(subtask)
#             elif task_type in ['analysis', 'generation']:
#                 return await self.execute_llm_task(subtask)
#             else:
#                 raise ValueError(f"Unknown task type: {task_type}")
                
#         except Exception as e:
#             logger.error(f"Error executing subtask {task_id}: {e}")
#             return {
#                 'task_id': task_id,
#                 'status': 'failed',
#                 'result': None,
#                 'error': str(e)
#             }

#     async def fetch_rag_context(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
#         """
#         Fetch context chunks from the custom RAG API.
#         """
#         payload = {
#             "query": query,
#             "collection": self.collection,
#             "top_k": top_k,
#         }

#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     self.api_base_url,
#                     json=payload,
#                     headers={"accept": "application/json", "Content-Type": "application/json"},
#                     timeout=60
#                 ) as response:
#                     if response.status == 200:
#                         data = await response.json()
#                         print(f"Rag data")
#                         return data.get("top_chunks", [])
#                     else:
#                         error_text = await response.text()
#                         raise Exception(f"RAG API error {response.status}: {error_text}")
#         except Exception as e:
#             logger.error(f"Error fetching RAG context: {e}")
#             return []
    
#     async def execute_web_search_task(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute web search task using Serper.dev"""
#         task_id = subtask['task_id']
#         parameters = subtask.get('parameters', {})
#         query = parameters.get('query', '')
#         num_results = parameters.get('num_results', 10)
#         search_type = parameters.get('search_type', 'search')  # search, images, news, places
#         country = parameters.get('country', 'us')
#         language = parameters.get('language', 'en')
        
#         if not self.serper_api_key:
#             return {
#                 'task_id': task_id,
#                 'status': 'failed',
#                 'result': None,
#                 'error': 'SERPER_API_KEY not configured'
#             }
        
#         if not query:
#             return {
#                 'task_id': task_id,
#                 'status': 'failed',
#                 'result': None,
#                 'error': 'Query parameter is required for web search'
#             }
        
#         # Check cache first
#         cache_key = f"web_search_{hash(f'{query}_{num_results}_{search_type}_{country}_{language}')}"
#         cached_result = cache.get(cache_key)
#         if cached_result:
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': cached_result,
#                 'cached': True
#             }
        
#         try:
#             search_results = await self._perform_serper_search(
#                 query=query,
#                 num_results=num_results,
#                 search_type=search_type,
#                 country=country,
#                 language=language
#             )
            
#             # Process and enhance results if needed
#             processed_results = await self._process_search_results(search_results, subtask)
            
#             # Cache the result
#             cache.set(cache_key, processed_results, ttl=3600)  # Cache for 1 hour
            
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': processed_results,
#                 'search_metadata': {
#                     'query': query,
#                     'num_results': len(processed_results.get('organic', [])),
#                     'search_type': search_type,
#                     'country': country,
#                     'language': language
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Web search task {task_id} failed: {e}")
#             raise
    
#     async def _perform_serper_search(
#         self, 
#         query: str, 
#         num_results: int = 10,
#         search_type: str = 'search',
#         country: str = 'us',
#         language: str = 'en'
#     ) -> Dict[str, Any]:
#         """Perform search using Serper.dev API"""
        
#         url = f"https://google.serper.dev/{search_type}"
        
#         payload = {
#             'q': query,
#             'num': num_results,
#             'gl': country,
#             'hl': language
#         }
        
#         headers = {
#             'X-API-KEY': self.serper_api_key,
#             'Content-Type': 'application/json'
#         }
        
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, json=payload, headers=headers) as response:
#                 if response.status == 200:
#                     return await response.json()
#                 else:
#                     error_text = await response.text()
#                     raise Exception(f"Serper API error {response.status}: {error_text}")
    
#     async def _process_search_results(self, search_results: Dict[str, Any], subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Process and optionally enhance search results"""
#         parameters = subtask.get('parameters', {})
#         enhance_with_llm = parameters.get('enhance_with_llm', False)
#         summarize_results = parameters.get('summarize_results', False)
        
#         processed_results = search_results.copy()
        
#         # Add metadata and clean up results
#         if 'organic' in processed_results:
#             for i, result in enumerate(processed_results['organic']):
#                 result['result_index'] = i + 1
#                 result['retrieved_at'] = asyncio.get_event_loop().time()
        
#         # Optionally enhance results with LLM analysis
#         if enhance_with_llm and 'organic' in processed_results:
#             try:
#                 enhanced_results = await self._enhance_results_with_llm(
#                     processed_results['organic'][:5],  # Enhance top 5 results
#                     subtask.get('llm_provider', 'openai')
#                 )
#                 processed_results['llm_enhanced'] = enhanced_results
#             except Exception as e:
#                 logger.warning(f"Failed to enhance results with LLM: {e}")
        
#         # Optionally summarize results
#         if summarize_results:
#             try:
#                 summary = await self._summarize_search_results(
#                     processed_results,
#                     subtask.get('llm_provider', 'anthropic')
#                 )
#                 processed_results['summary'] = summary
#             except Exception as e:
#                 logger.warning(f"Failed to summarize results: {e}")
        
#         return processed_results
    
#     async def _enhance_results_with_llm(self, results: List[Dict], llm_provider: str) -> List[Dict]:
#         """Enhance search results with LLM analysis"""
#         llm = self.get_llm(llm_provider)
#         enhanced_results = []
        
#         for result in results:
#             try:
#                 # Create analysis prompt
#                 prompt = f"""
#                 Analyze this search result and provide key insights:
                
#                 Title: {result.get('title', 'N/A')}
#                 Snippet: {result.get('snippet', 'N/A')}
#                 URL: {result.get('link', 'N/A')}
                
#                 Provide:
#                 1. Main topic/theme
#                 2. Key information points
#                 3. Relevance score (1-10)
#                 4. Content type (news, academic, commercial, etc.)
                
#                 Format as JSON.
#                 """
                
#                 messages = [
#                     SystemMessage(content="You are analyzing search results to extract key insights."),
#                     HumanMessage(content=prompt)
#                 ]
                
#                 response = await asyncio.to_thread(llm, messages)
#                 analysis = response.content if hasattr(response, 'content') else str(response)
                
#                 enhanced_result = result.copy()
#                 enhanced_result['llm_analysis'] = analysis
#                 enhanced_results.append(enhanced_result)
                
#             except Exception as e:
#                 logger.warning(f"Failed to analyze result {result.get('title', 'unknown')}: {e}")
#                 enhanced_results.append(result)
        
#         return enhanced_results
    
#     async def _summarize_search_results(self, search_results: Dict[str, Any], llm_provider: str) -> str:
#         """Generate a summary of search results using LLM"""
#         llm = self.get_llm(llm_provider)
        
#         # Extract key information from results
#         organic_results = search_results.get('organic', [])[:10]  # Top 10 results
        
#         results_text = "\n\n".join([
#             f"Result {i+1}:\nTitle: {result.get('title', 'N/A')}\nSnippet: {result.get('snippet', 'N/A')}"
#             for i, result in enumerate(organic_results)
#         ])
        
#         prompt = f"""
#         Summarize these search results in a comprehensive but concise manner:
        
#         {results_text}
        
#         Provide:
#         1. Main themes and topics covered
#         2. Key findings and information
#         3. Different perspectives or viewpoints (if any)
#         4. Overall relevance and quality of results
        
#         Keep the summary informative but concise (300-500 words).
#         """
        
#         messages = [
#             SystemMessage(content="You are summarizing search results to provide key insights."),
#             HumanMessage(content=prompt)
#         ]
        
#         response = await asyncio.to_thread(llm, messages)
#         return response.content if hasattr(response, 'content') else str(response)
    
#     async def execute_rag_task(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute RAG (Retrieval-Augmented Generation) task"""
#         task_id = subtask['task_id']
#         parameters = subtask.get('parameters', {})
#         query = parameters.get('query', '')
#         collection = 'sandbox'
        
#         # Check cache first
#         cache_key = f"rag_{hash(query)}"
#         cached_result = cache.get(cache_key)
#         if cached_result:
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': cached_result,
#                 'cached': True
#             }
        
#         try:
#             # Fetch RAG context
#             context_results = await self.fetch_rag_context(query, top_k=10)
            
#             if not context_results:
#                 return {
#                     'task_id': task_id,
#                     'status': 'completed',
#                     'result': "No relevant context found for the query.",
#                     'context': []
#                 }
            
#             # Prepare context for LLM
#             context_text = "\n\n".join([
#                 f"Source: {result.get('metadata', {}).get('file_id', 'unknown')}\n{result['text']}"
#                 for result in context_results
#             ])
            
#             sources = [result.get('metadata', {}) for result in context_results]
            
#             # Generate response using RAG prompt
#             prompt = self.prompts.get_prompt(
#                 'rag_retrieval',
#                 query=query,
#                 context=context_text,
#                 sources=sources
#             )
            
#             # Use specified LLM provider
#             llm_provider = subtask.get('llm_provider', 'openai')
#             llm = self.get_llm(llm_provider)
            
#             messages = [
#                 SystemMessage(content="You are a helpful assistant performing RAG retrieval."),
#                 HumanMessage(content=prompt)
#             ]
            
#             response = await asyncio.to_thread(llm, messages)
#             result = response.content if hasattr(response, 'content') else str(response)
            
#             # Cache the result
#             cache.set(cache_key, result)
            
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': result,
#                 'context': context_results,
#                 'llm_provider': llm_provider
#             }
            
#         except Exception as e:
#             logger.error(f"RAG task {task_id} failed: {e}")
#             raise
    
#     async def execute_rat_task(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute RAT (Retrieval-Augmented Thinking) task"""
#         task_id = subtask['task_id']
#         parameters = subtask.get('parameters', {})
#         query = parameters.get('query', '')
#         context = parameters.get('context', '')
#         collection = 'sandbox'
        
#         try:
#             # If no context provided, fetch it
#             if not context:
#                 context_results = await self.fetch_rag_context(query,top_k=10)
#                 context = "\n\n".join([result['text'] for result in context_results])
            
#             # Generate reasoning using RAT prompt
#             prompt = self.prompts.get_prompt(
#                 'rat_reasoning',
#                 query=query,
#                 context=context,
#                 task_description=subtask.get('description', '')
#             )
            
#             # Use specified LLM provider
#             llm_provider = subtask.get('llm_provider', 'anthropic')  # Default to Claude for reasoning
#             llm = self.get_llm(llm_provider)
            
#             messages = [
#                 SystemMessage(content="You are an expert reasoner performing analytical thinking."),
#                 HumanMessage(content=prompt)
#             ]
            
#             response = await asyncio.to_thread(llm, messages)
#             result = response.content if hasattr(response, 'content') else str(response)
            
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': result,
#                 'reasoning_context': context,
#                 'llm_provider': llm_provider
#             }
            
#         except Exception as e:
#             logger.error(f"RAT task {task_id} failed: {e}")
#             raise
    
#     async def execute_llm_task(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute general LLM task (analysis, generation, etc.)"""
#         task_id = subtask['task_id']
#         parameters = subtask.get('parameters', {})
#         query = parameters.get('query', '')
        
#         try:
#             # Use specified LLM provider
#             llm_provider = subtask.get('llm_provider', 'openai')
#             llm = self.get_llm(llm_provider)
            
#             messages = [
#                 SystemMessage(content=f"You are performing: {subtask.get('description', 'analysis')}"),
#                 HumanMessage(content=query)
#             ]
            
#             response = await asyncio.to_thread(llm, messages)
#             result = response.content if hasattr(response, 'content') else str(response)
            
#             return {
#                 'task_id': task_id,
#                 'status': 'completed',
#                 'result': response
#                 }

#         except Exception as e:
#             logger.error(f"LLM task {task_id} failed: {e}")
#             raise


# tool_executor = ToolExecutor()