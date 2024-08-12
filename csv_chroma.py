import pandas as pd
import uuid
from tqdm.auto import tqdm
import logging
import sys
import colorlog
import chromadb
from chromadb_mistral_embeddings import MistralEmbeddingFunction
from dotenv import load_dotenv
from mistralai.client import MistralClient
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

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chromadb_csv", settings=Settings(anonymized_telemetry=False))

# 1. Data Preparation
def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, dtype=str)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, dtype=str, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    def safe_numeric(x):
        try:
            return pd.to_numeric(x)
        except:
            return x
    
    logger.info("Processing columns...")
    for col in tqdm(df.columns, desc="Converting to numeric"):
        df[col] = df[col].apply(safe_numeric)
    
    def safe_eval(x):
        try:
            return eval(x)
        except:
            return x

    for col in tqdm(df.columns, desc="Processing list-like strings"):
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ''
            if sample.startswith('[') and sample.endswith(']'):
                df[col] = df[col].apply(safe_eval)

    logger.info("Data loaded and processed successfully.")
    return df

# 2. Indexing with Chroma DB
def create_chroma_index(df, collection_name):
    logger.info(f"Creating Chroma DB index with collection name: {collection_name}...")
    
    # Create or get a collection
    collection = chroma_client.create_collection(name=collection_name)
    
    # Create Mistral embedding function
    mistral_ef = MistralEmbeddingFunction(api_key=client._api_key, model_name="mistral-embed")
    
    # Check if 'id' column exists, if not, create it from the index
    if 'id' not in df.columns:
        logger.info("'id' column not found. Creating 'id' column from index.")
        df['id'] = df.index.astype(str)
    
    # Prepare data for indexing
    documents = df.astype(str).agg(' '.join, axis=1).tolist()
    
    # Convert all metadata to strings and ensure unique IDs
    metadatas = df.map(lambda x: str(x) if isinstance(x, list) else x).to_dict('records')
    
    # Generate unique IDs
    unique_ids = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Add original ID to metadata
    for i, metadata in enumerate(metadatas):
        metadata['original_id'] = df.iloc[i]['id']
    
    logger.info("Adding data to Chroma DB...")
    with tqdm(total=len(unique_ids), desc="Adding to Chroma DB", unit="batch", leave=False) as pbar:
        for i in range(0, len(unique_ids), 100):  # Process in batches of 100
            batch_ids = unique_ids[i:i+100]
            batch_documents = documents[i:i+100]
            batch_metadatas = metadatas[i:i+100]
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=mistral_ef(batch_documents)
            )
            pbar.update(len(batch_ids))
    
    logger.info("Chroma DB index created successfully.")
    return collection

# Main Ingestion Function
def ingest_data(csv_file):
    logger.info("Starting data ingestion process...")
    
    # Ask user for collection name
    collection_name = input("Please enter a name for the Chroma DB collection: ")
    
    # Load and index data
    df = load_data(csv_file)
    
    # Create Chroma DB index
    collection = create_chroma_index(df, collection_name)
    
    logger.info("Data ingestion completed successfully.")
    return collection

# Usage example
csv_file = "USA SALES DATA UPDATED - JATO.xlsx"
ingest_data(csv_file)
