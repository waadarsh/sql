# retrieval_script.py

import logging
import sys
from tqdm.auto import tqdm
import colorlog
import chromadb
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from chromadb_mistral_embeddings import MistralEmbeddingFunction
from chromadb.config import Settings

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

# Set your Mistral AI API key
load_dotenv('.env')
client = MistralClient()

# Create Mistral embedding function
mistral_ef = MistralEmbeddingFunction(api_key=client._api_key, model_name="mistral-embed")

# Initialize Chroma client and load the collection
chroma_client = chromadb.PersistentClient(path="./chromadb_csv", settings=Settings(anonymized_telemetry=False))

# 3. Retrieval using Chroma DB
def retrieve_relevant_rows(query, collection, embedding_function, top_k=5):
    logger.info(f"Retrieving top {top_k} relevant rows...")
    results = collection.query(
        n_results=top_k,
        query_embeddings=embedding_function([query])
    )
    return results

# 4. Augmentation
def augment_query(query, relevant_results):
    logger.info("Augmenting query with relevant data...")
    context = "\n".join([f"Row {i+1}: " + str(metadata) for i, metadata in enumerate(relevant_results['metadatas'][0])])
    augmented_input = f"Query: {query}\n\nRelevant Data:\n{context}\n\nBased on the above information, please answer the query."
    return augmented_input

# 5. Generation using Mistral AI
def generate_response(augmented_input):
    logger.info("Generating response using Mistral AI...")
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that answers questions based on the provided data."),
        ChatMessage(role="user", content=augmented_input)
    ]
    
    chat_response = client.chat(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.5,
    )
    
    return chat_response.choices[0].message.content

def rag_with_csv_mistral_test(query, collection_name, columns=None):
    logger.info(f"Starting RAG pipeline for query: '{query}'")
    
    # Create Mistral embedding function
    mistral_ef = MistralEmbeddingFunction(api_key=client._api_key, model_name="mistral-embed")
    
    # Get the collection
    collection = chroma_client.get_collection(name=collection_name, embedding_function=mistral_ef)
    
    # Get a sample document to extract column names
    sample_results = collection.peek(limit=1)
    if sample_results['metadatas'] and sample_results['metadatas'][0]:
        columns = list(sample_results['metadatas'][0].keys())
        primary_column = columns[0] 
    else:
        logger.error("Unable to retrieve column information from the collection.")
        return "Error: Unable to process the query due to missing column information."

    # Split the query into individual items
    prefix = f"give {primary_column} of"
    if prefix.lower() in query.lower():
        query = query.lower().replace(prefix.lower(), "").strip()
    items = [item.strip() for item in query.split("and")]
    
    all_results = []
    for item in items:
        results = retrieve_relevant_rows(item, collection, mistral_ef, top_k=5)
        all_results.extend(results['metadatas'][0])
    
    # Augment the query
    augmented_input = f"""Query: {query}

        Relevant Data:
        {chr(10).join([f"Row {i+1}: {metadata}" for i, metadata in enumerate(all_results)])}

        Available Columns: {', '.join(columns)}

        Instructions:
        1. Use the provided data to answer the query as accurately as possible.
        2. If information is not found in the data, explicitly state that no information was found for that Query.
        3. Provide context and explanations where appropriate.
        4. Use the available columns to structure your response.

        Based on the above information and instructions, please answer the query."""
    
    # Generate response using Mistral AI
    response = generate_response(augmented_input)
    
    logger.info("RAG pipeline completed successfully.")
    return response

def get_valid_collection():
    while True:
        collection_name = input("Enter the collection name: ")
        try:
            collection = chroma_client.get_collection(name=collection_name)
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
                result = rag_with_csv_mistral_test(query, collection_name)
                print("\nGenerated Response:")
                print(result)
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"An error occurred while processing your query: {str(e)}")
                print("Please try again or change to a different collection.")

if __name__ == "__main__":
    logger.info("Starting RAG chatbot...")
    chat_loop()
