import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any

# Load environment variables
load_dotenv('.env')

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chromadb_csv")

# Create OpenAI embedding function
openai_ef = OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_EMB_ENDPOINT_URI"),
    api_type="azure",
    api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"),
)

def retrieve_documents(query: str, collection_name: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents from Chroma DB based on a query.
    
    Args:
    query (str): The search query.
    collection_name (str): Name of the Chroma DB collection to search.
    n_results (int): Number of results to return.
    
    Returns:
    List[Dict[str, Any]]: List of dictionaries containing retrieved documents and their metadata.
    """
    # Get the collection
    collection = chroma_client.get_collection(name=collection_name)
    
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process and format the results
    formatted_results = []
    for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        formatted_results.append({
            "document": doc,
            "metadata": metadata,
            "distance": distance
        })
    
    return formatted_results

def display_results(results: List[Dict[str, Any]]):
    """
    Display the retrieved results in a formatted manner.
    
    Args:
    results (List[Dict[str, Any]]): List of retrieved documents and their metadata.
    """
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Distance: {result['distance']:.4f}")
        print("Document:")
        print(result['document'])
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        print("-" * 50)

def main():
    # Ask user for collection name
    collection_name = input("Enter the name of the Chroma DB collection to search: ")
    
    while True:
        # Get user query
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        # Retrieve documents
        results = retrieve_documents(query, collection_name)
        
        # Display results
        display_results(results)

if __name__ == "__main__":
    main()
