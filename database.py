import os
from typing import Dict, List, Any, Optional
import pymongo
from pymongo import MongoClient
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import logging
from utils import logger
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class DatabaseHandler:
    def __init__(self):
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.db_name = os.getenv('MONGO_DB_NAME', 'brain_heart_db')
        self.client = None
        self.db = None
        self.faiss_index = None
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model = None
        self.embedding_dimension = 384
        
        # Collections
        self.files_collection = None
        self.prompts_collection = None
        self.embeddings_collection = None
        
        self._connect()
        self._initialize_faiss()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            
            # Initialize collections
            self.files_collection = self.db.files
            self.prompts_collection = self.db.prompts
            self.embeddings_collection = self.db.embeddings
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            # Use in-memory fallback
            self._initialize_memory_fallback()
    
    def _initialize_memory_fallback(self):
        """Initialize in-memory storage as fallback"""
        logger.info("Using in-memory storage as fallback")
        self.memory_storage = {
            'files': {},
            'prompts': {},
            'embeddings': {}
        }
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        try:
            # Create a new FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner Product similarity
            logger.info("FAISS index initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
    
    def save_file(self, filename: str, content: bytes, file_type: str, metadata: Dict = None) -> str:
        """Save uploaded file to database"""
        try:
            file_doc = {
                'filename': filename,
                'content': content,
                'file_type': file_type,
                'metadata': metadata or {},
                'uploaded_at': pymongo.datetime.datetime.utcnow()
            }
            
            if self.files_collection:
                result = self.files_collection.insert_one(file_doc)
                file_id = str(result.inserted_id)
            else:
                # Memory fallback
                file_id = f"file_{len(self.memory_storage['files'])}"
                self.memory_storage['files'][file_id] = file_doc
            
            logger.info(f"File {filename} saved with ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            raise
    
    def get_file(self, file_id: str) -> Optional[Dict]:
        """Retrieve file by ID"""
        try:
            if self.files_collection:
                return self.files_collection.find_one({'_id': pymongo.ObjectId(file_id)})
            else:
                return self.memory_storage['files'].get(file_id)
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}")
            return None
    
    def save_embedding(self, text: str, embedding: np.ndarray, metadata: Dict = None) -> int:
        """Save text embedding to FAISS and metadata to MongoDB"""
        try:
            # Add to FAISS index
            embedding_2d = embedding.reshape(1, -1).astype('float32')
            self.faiss_index.add(embedding_2d)
            embedding_id = self.faiss_index.ntotal - 1
            
            # Save metadata to MongoDB
            embedding_doc = {
                'embedding_id': embedding_id,
                'text': text,
                'metadata': metadata or {},
                'created_at': pymongo.datetime.datetime.utcnow()
            }
            
            if self.embeddings_collection:
                self.embeddings_collection.insert_one(embedding_doc)
            else:
                self.memory_storage['embeddings'][embedding_id] = embedding_doc
            
            return embedding_id
            
        except Exception as e:
            logger.error(f"Error saving embedding: {e}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar embeddings using FAISS"""
        try:
            if self.faiss_index.ntotal == 0:
                return []
            
            query_2d = query_embedding.reshape(1, -1).astype('float32')
            similarities, indices = self.faiss_index.search(query_2d, min(top_k, self.faiss_index.ntotal))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                # Get metadata from MongoDB
                if self.embeddings_collection:
                    embedding_doc = self.embeddings_collection.find_one({'embedding_id': int(idx)})
                else:
                    embedding_doc = self.memory_storage['embeddings'].get(int(idx))
                
                if embedding_doc:
                    results.append({
                        'text': embedding_doc['text'],
                        'similarity': float(similarity),
                        'metadata': embedding_doc.get('metadata', {}),
                        'embedding_id': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dimension)
    
    def save_prompt(self, prompt_type: str, prompt_content: str, metadata: Dict = None):
        """Save prompt to database"""
        try:
            prompt_doc = {
                'type': prompt_type,
                'content': prompt_content,
                'metadata': metadata or {},
                'updated_at': pymongo.datetime.datetime.utcnow()
            }
            
            if self.prompts_collection:
                self.prompts_collection.replace_one(
                    {'type': prompt_type}, 
                    prompt_doc, 
                    upsert=True
                )
            else:
                self.memory_storage['prompts'][prompt_type] = prompt_doc
            
            logger.info(f"Prompt {prompt_type} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving prompt {prompt_type}: {e}")
            raise
    
    def get_prompt(self, prompt_type: str) -> Optional[str]:
        """Retrieve prompt by type"""
        try:
            if self.prompts_collection:
                prompt_doc = self.prompts_collection.find_one({'type': prompt_type})
            else:
                prompt_doc = self.memory_storage['prompts'].get(prompt_type)
            
            return prompt_doc['content'] if prompt_doc else None
            
        except Exception as e:
            logger.error(f"Error retrieving prompt {prompt_type}: {e}")
            return None
    
    def get_all_files(self) -> List[Dict]:
        """Get all uploaded files"""
        try:
            if self.files_collection:
                return list(self.files_collection.find({}, {'content': 0}))  # Exclude content for performance
            else:
                return [
                    {**doc, 'content': None} for doc in self.memory_storage['files'].values()
                ]
        except Exception as e:
            logger.error(f"Error retrieving files: {e}")
            return []
    
    def index_file_content(self, file_id: str, content_chunks: List[str]):
        """Index file content for RAG retrieval"""
        try:
            for i, chunk in enumerate(content_chunks):
                embedding = self.embed_text(chunk)
                self.save_embedding(
                    text=chunk,
                    embedding=embedding,
                    metadata={
                        'file_id': file_id,
                        'chunk_index': i,
                        'source': 'file_upload'
                    }
                )
            logger.info(f"Indexed {len(content_chunks)} chunks from file {file_id}")
        except Exception as e:
            logger.error(f"Error indexing file content: {e}")
            raise

# Global database handler
db_handler = DatabaseHandler()