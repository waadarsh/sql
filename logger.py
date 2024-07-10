import os
import logging
import colorlog
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import (
    Collection,
    connections,
    WeightedRanker,
)

# Logging configuration
log_format = ("%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s")
formatter = colorlog.ColoredFormatter(
    log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

CONNECTION_URI = "http://localhost:19530"

# Connect to Milvus
connections.connect(uri=CONNECTION_URI)

# Embedding functions
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
dense_embedding_func = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Retrieve documents from collection
collection_name = "Nissan_Ariya"
collection = Collection(collection_name)
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"

sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}
retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    rerank=WeightedRanker(0.5, 0.5),
    anns_fields=[dense_field, sparse_field],
    field_embeddings=[dense_embedding_func, dense_embedding_func],
    field_search_params=[dense_search_params, sparse_search_params],
    top_k=3,
    text_field=text_field,
)

# Example query
query = "What are the story about ventures?"
logger.debug("Invoking hybrid search")
results = retriever.invoke(query)
logger.info("Retriever invoked successfully")
for result in results:
    print(result)
