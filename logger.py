import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import colorlog
import logging

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# replace
ZILLIZ_CLOUD_URI = "https://in03-d274f3928840065.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_CLOUD_USERNAME = 'db_d274f3928840065'
ZILLIZ_CLOUD_PASSWORD = 'Fu8[^*[rAXb0DX&v'

# Define a custom log format with colors
log_format = ("%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a colored formatter
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

# Configure the logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# logger.debug("Attempting to retrieve web document")
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# logger.info("Document loaded successfully")
# splits = text_splitter.split_documents(docs)
# logger.info("Documents split into chunks")

logger.debug("Attempting to load pdf document")
loader = PyPDFLoader("2023-nissan-ariya-en.pdf", extract_images=True)
pages = loader.load()
logger.info("Pdf loaded Successfully")
splits_1 = text_splitter.split_documents(pages)
logger.info("pdf split into chunks")

# logger.debug("Attempting to vectorize document")
# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
# logger.info("document stored in chroma db")

logger.debug("Attempting to vectorize pdf")
# vectorstore = Milvus.from_documents(
#     splits_1,
#     embedding_function,
#     connection_args={"uri": "http://localhost:19530"},
# )

vectorstore = Milvus.from_documents(
    splits_1,
    embedding_function,
    connection_args={
        "uri": ZILLIZ_CLOUD_URI,
        "user": ZILLIZ_CLOUD_USERNAME,
        "password": ZILLIZ_CLOUD_PASSWORD,
        "secure": True,
    },
)

logger.info("pdf stored in milvus db")
