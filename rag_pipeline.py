"""
RAG Pipeline Module
Handles document ingestion, embedding, indexing, retrieval, and generation.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec


class RAGPipeline:
    """Complete RAG pipeline for document Q&A."""
    
    def __init__(
        self,
        docs_dir: str = "./docs",
        index_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            docs_dir: Directory containing documents
            index_name: Name of Pinecone index (defaults to PINECONE_INDEX_NAME env var or "rag-assessment")
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        self.docs_dir = Path(docs_dir)
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-assessment")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Vector store (will be initialized in setup_index)
        self.vector_store: Optional[PineconeVectorStore] = None
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the docs directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not self.docs_dir.exists():
            raise ValueError(f"Directory {self.docs_dir} does not exist")
        
        # Load all .txt and .md files
        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path.name)}
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        for file_path in self.docs_dir.glob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path.name)}
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        if not documents:
            raise ValueError(f"No documents found in {self.docs_dir}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
        return chunks
    
    def index_has_documents(self) -> bool:
        """
        Check if the index has any documents.
        
        Returns:
            True if index has documents, False otherwise
        """
        try:
            pc = Pinecone()
            index = pc.Index(self.index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            return total_vectors > 0
        except Exception as e:
            print(f"Warning: Could not check index stats: {e}")
            return False
    
    def setup_index(self, force_recreate: bool = False) -> None:
        """
        Set up Pinecone index and vector store.
        
        Args:
            force_recreate: If True, delete and recreate the index
        """
        pc = Pinecone()
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            if force_recreate:
                print(f"Deleting existing index: {self.index_name}")
                pc.delete_index(self.index_name)
            else:
                print(f"Using existing index: {self.index_name}")
                self.vector_store = PineconeVectorStore.from_existing_index(
                    index_name=self.index_name,
                    embedding=self.embeddings
                )
                # Check if index has documents
                if not self.index_has_documents():
                    print("Warning: Index exists but appears to be empty. Consider using --reindex to populate it.")
                return
        
        # Create new index if it doesn't exist
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            pc.create_index(
                name=self.index_name,
                dimension=1536,  # text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
                )
            )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
    
    def ingest_documents(self, force_recreate: bool = False) -> None:
        """
        Complete ingestion pipeline: load, split, embed, and index documents.
        
        Args:
            force_recreate: If True, recreate the index before ingesting
        """
        print("Starting document ingestion...")
        
        # Setup index
        self.setup_index(force_recreate=force_recreate)
        
        # Load documents
        documents = self.load_documents()
        
        # Split documents
        chunks = self.split_documents(documents)
        
        # Add to vector store
        print(f"Adding {len(chunks)} chunks to vector store...")
        try:
            # Add documents in batches for better reliability
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.vector_store.add_documents(batch)
                if i + batch_size < len(chunks):
                    print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
            
            # Wait a moment for indexing to complete
            import time
            time.sleep(2)
            
            print("Document ingestion complete!")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def create_qa_chain(self):
        """
        Create a QA chain with custom prompt using LCEL.
        
        Returns:
            Retrieval chain
        """
        # Custom prompt template - more lenient to encourage answers
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use the context below to answer the question. If the context contains relevant information, provide a clear and helpful answer.
Only say "I don't know" if the context truly doesn't contain any relevant information about the question.

Context:
{context}

Question: {question}

Provide a helpful answer based on the context above:"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the chain using LCEL
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return qa_chain
    
    def answer_question(self, question: str, debug: bool = False) -> dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            debug: If True, print debug information about retrieved documents
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Run ingest_documents() first.")
        
        # Create retriever for getting sources
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        # Get relevant documents for source attribution
        docs = retriever.invoke(question)
        
        if debug:
            print(f"\n[DEBUG] Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"     Content preview: {doc.page_content[:100]}...")
        
        # Check if we got any documents
        if not docs:
            return {
                "answer": "I don't know. No relevant documents were found in the knowledge base.",
                "sources": []
            }
        
        # Create QA chain
        qa_chain = self.create_qa_chain()
        
        # Get answer
        answer = qa_chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": [
                doc.metadata.get("source", "Unknown")
                for doc in docs
            ]
        }
