# retrieval_script.py

import logging
import os
import sys
import colorlog
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from chromadb.config import Settings
from typing import Dict, Any

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up logger
def setup_logger():
    logger = colorlog.getLogger()
    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler(stream=sys.stderr)
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# Set your OpenAI API key
load_dotenv('.env')
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

# Create OpenAI embedding function
openai_ef = OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_EMB_ENDPOINT_URI"),
    api_type="azure",
    api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"),
)

# Initialize Chroma client and load the collection
chroma_client = chromadb.PersistentClient(path="./chromadb_csv", settings=Settings(anonymized_telemetry=False))

# 3. Retrieval using Chroma DB
def retrieve_relevant_rows(query: str, collection, embedding_function, top_k: int = 5) -> Dict[str, Any]:
    logger.info(f"Retrieving top {top_k} relevant rows for query: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    
    return {
        'ids': results['ids'][0],
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0]
    }

# 4. Augmentation
def augment_query(query, relevant_results):
    logger.info("Augmenting query with relevant data...")
    context = "\n".join([f"Row {i+1}: " + str(metadata) for i, metadata in enumerate(relevant_results['metadatas'])])
    augmented_input = f"Query: {query}\n\nRelevant Data:\n{context}\n\nBased on the above information, please answer the query."
    return augmented_input

# 5. Generation using Azure OpenAI
def generate_response(augmented_input):
    logger.info("Generating response using Azure OpenAI...")
    messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided data.",
            },
            {"role": "user", "content": augmented_input},
        ]
    
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
    )
    
    return chat_response.choices[0].message.content

def rag_with_csv_openai_test(query, collection_name, columns=None):
    logger.info(f"Starting RAG pipeline for query: '{query}'")
    
    # Get the collection
    collection = chroma_client.get_collection(name=collection_name, embedding_function=openai_ef)
    
    # Get a sample document to extract column names
    sample_results = collection.peek(limit=1)
    if sample_results['metadatas'] and sample_results['metadatas'][0]:
        columns = list(sample_results['metadatas'][0].keys())
        primary_column = columns[0] 
    else:
        logger.error("Unable to retrieve column information from the collection.")
        return "Error: Unable to process the query due to missing column information."

    # Retrieve relevant rows
    results = retrieve_relevant_rows(query, collection, openai_ef, top_k=5)
    
    # Augment the query
    augmented_input = f"""Query: {query}

        Relevant Data:
        {chr(10).join([f"Row {i+1}: {metadata}" for i, metadata in enumerate(results['metadatas'])])}

        Available Columns: {', '.join(columns)}

        Instructions:
        1. Use the provided data to answer the query as accurately as possible.
        2. If information is not found in the data, explicitly state that no information was found for that Query.
        3. Provide context and explanations where appropriate.
        4. Use the available columns to structure your response.

        Based on the above information and instructions, please answer the query."""
    
    # Generate response using Azure OpenAI
    response = generate_response(augmented_input)
    
    logger.info("RAG pipeline completed successfully.")
    return response

def get_valid_collection():
    while True:
        collection_name = input("Enter the collection name: ")
        try:
            collection = chroma_client.get_collection(name=collection_name)  # noqa: F841
            return collection_name
        except ValueError:
            print(f"Collection '{collection_name}' does not exist. Please check the spelling or enter a valid collection name.")

def chat_loop():
    collection_name = get_valid_collection()
    
    while True:
        query = input(f"Enter your query for collection '{collection_name}' (or 'change collection' to switch, 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        elif query.lower() == 'change collection':
            collection_name = get_valid_collection()
            print(f"Switched to collection: {collection_name}")
        else:
            try:
                result = rag_with_csv_openai_test(query, collection_name)
                print("\nGenerated Response:")
                print(result)
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"An error occurred while processing your query: {str(e)}")
                print("Please try again or change to a different collection.")

if __name__ == "__main__":
    logger.info("Starting RAG chatbot...")
    chat_loop()
