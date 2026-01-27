#!/usr/bin/env python3
"""
Command-line Q&A Assistant
A RAG-based question answering system for internal documentation.
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline


def main():
    """Main CLI entry point."""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in a .env file or as environment variables.")
        sys.exit(1)
    
    # Get default index name from environment or use fallback
    default_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-assessment")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Q&A Assistant for internal documentation using RAG"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="The question to answer"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./docs",
        help="Directory containing documents (default: ./docs)"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=default_index_name,
        help=f"Pinecone index name (default: {default_index_name} from PINECONE_INDEX_NAME env var)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing of documents (deletes existing index)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information about retrieved documents"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG pipeline
        print("Initializing RAG pipeline...")
        pipeline = RAGPipeline(
            docs_dir=args.docs_dir,
            index_name=args.index_name,
            top_k=args.top_k
        )
        
        # Check if index exists and has data, or if reindex is requested
        from pinecone import Pinecone
        pc = Pinecone()
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if args.reindex or args.index_name not in existing_indexes:
            # Ingest documents
            pipeline.ingest_documents(force_recreate=args.reindex)
        else:
            # Just setup the existing index
            pipeline.setup_index(force_recreate=False)
            # Check if index is empty and needs ingestion
            if not pipeline.index_has_documents():
                print("Index is empty. Ingesting documents...")
                pipeline.ingest_documents(force_recreate=False)
        
        # Answer the question
        print(f"\nQuestion: {args.question}\n")
        result = pipeline.answer_question(args.question, debug=args.debug)
        
        # Display answer
        print("Answer:", result["answer"])
        
        # Display sources if available
        if result.get("sources"):
            print(f"\nSources: {', '.join(set(result['sources']))}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
