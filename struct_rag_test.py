import logging
import os
import sys
import json
import colorlog
import chromadb
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings
from typing import Dict, Any, Optional, List
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone
from pony.orm import Database, Required, Optional as PonyOptional, Json, Set, db_session, commit
from pony.orm.asynchronous import set_sql_debug
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv('.env')

# Database setup
db = Database()

class ChatSession(db.Entity):
    """
    Represents a chat session in the database.
    
    Attributes:
        id (uuid.UUID): Unique identifier for the session.
        created_at (datetime): Timestamp when the session was created.
        updated_at (datetime): Timestamp when the session was last updated.
        chat_history (Json): JSON representation of the chat history.
    """
    id = Required(uuid.UUID, default=uuid.uuid4, primary_key=True)
    created_at = Required(datetime, default=lambda: datetime.now(timezone.utc))
    updated_at = Required(datetime, default=lambda: datetime.now(timezone.utc))
    chat_history = Required(Json)

# Connect to the database
db.bind(provider='postgres', host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), 
        password=os.getenv('DB_PASSWORD'), database=os.getenv('DB_NAME'))
db.generate_mapping(create_tables=True)

set_sql_debug(True)  # Enable SQL debug logging

class RAGChatbot:
    """
    Retrieval-Augmented Generation (RAG) Chatbot class.
    
    This class handles the main logic for the chatbot, including data retrieval,
    response generation, and interaction management.
    """

    def __init__(self, client: AsyncAzureOpenAI, openai_ef: OpenAIEmbeddingFunction, chroma_client: chromadb.PersistentClient):
        """
        Initialize the RAGChatbot.
        
        Args:
            client (AsyncAzureOpenAI): Azure OpenAI client for text generation.
            openai_ef (OpenAIEmbeddingFunction): OpenAI embedding function.
            chroma_client (chromadb.PersistentClient): Chroma client for vector storage.
        """
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set up logger
        self.logger = self.setup_logger()

        # Initialize clients
        self.client = client
        self.openai_ef = openai_ef
        self.chroma_client = chroma_client

        # Load the collections
        self.excel_collection = self.chroma_client.get_collection(name="test", embedding_function=self.openai_ef)
        self.pdf_collection = self.chroma_client.get_collection(name="pdf", embedding_function=self.openai_ef)

    def setup_logger(self):
        """
        Set up and configure the logger for the application.
        
        Returns:
            logging.Logger: Configured logger instance.
        """
        # Logger setup code here (not provided in the original snippet)
        pass

    async def retrieve_excel_data(self, query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant data from the Excel collection based on the query.
        
        Args:
            query (str): The query to search for.
            top_k (int): Number of top results to retrieve.
        
        Returns:
            Optional[Dict[str, Any]]: Retrieved data or None if no results.
        """
        # Excel data retrieval logic here (not provided in the original snippet)
        pass

    async def retrieve_pdf_data(self, query: str, top_k: int = 5) -> Optional[list]:
        """
        Retrieve relevant data from the PDF collection based on the query.
        
        Args:
            query (str): The query to search for.
            top_k (int): Number of top results to retrieve.
        
        Returns:
            Optional[list]: Retrieved data or None if no results.
        """
        # PDF data retrieval logic here (not provided in the original snippet)
        pass

    async def concurrent_retrieval(self, query: str, top_k: int = 5) -> tuple:
        """
        Concurrently retrieve data from both Excel and PDF collections.
        
        Args:
            query (str): The query to search for.
            top_k (int): Number of top results to retrieve from each collection.
        
        Returns:
            tuple: A tuple containing the results from Excel and PDF retrievals.
        """
        # Concurrent retrieval logic here (not provided in the original snippet)
        pass

    def format_excel_data(self, excel_results: Optional[Dict[str, Any]]) -> str:
        """
        Format the retrieved Excel data into a string representation.
        
        Args:
            excel_results (Optional[Dict[str, Any]]): The retrieved Excel data.
        
        Returns:
            str: Formatted string representation of the Excel data.
        """
        # Excel data formatting logic here (not provided in the original snippet)
        pass

    def format_pdf_data(self, pdf_results: Optional[list]) -> str:
        """
        Format the retrieved PDF data into a string representation.
        
        Args:
            pdf_results (Optional[list]): The retrieved PDF data.
        
        Returns:
            str: Formatted string representation of the PDF data.
        """
        # PDF data formatting logic here (not provided in the original snippet)
        pass

    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format the chat history into a string representation.
        
        Args:
            chat_history (List[Dict[str, str]]): The chat history.
        
        Returns:
            str: Formatted string representation of the chat history.
        """
        # Chat history formatting logic here (not provided in the original snippet)
        pass

    async def generate_response(self, augmented_input: str) -> str:
        """
        Generate a response using the Azure OpenAI client.
        
        Args:
            augmented_input (str): The input prompt augmented with retrieved data.
        
        Returns:
            str: Generated response from the AI model.
        """
        # Response generation logic here (not provided in the original snippet)
        pass

    async def rag(self, query: str, chat_history: List[Dict[str, str]], columns: List[str] = None) -> str:
        """
        Perform the Retrieval-Augmented Generation process.
        
        Args:
            query (str): The user's query.
            chat_history (List[Dict[str, str]]): The chat history.
            columns (List[str], optional): Specific columns to retrieve from Excel data.
        
        Returns:
            str: Generated response based on the RAG process.
        """
        # RAG process logic here (not provided in the original snippet)
        pass

    async def process_message(self, websocket: WebSocket, message: str, session_id: uuid.UUID):
        """
        Process an incoming message from a WebSocket connection.
        
        Args:
            websocket (WebSocket): The WebSocket connection.
            message (str): The incoming message.
            session_id (uuid.UUID): The session ID for the current chat.
        """
        # Message processing logic here (not provided in the original snippet)
        pass

class ConnectionManager:
    """
    Manages WebSocket connections for the application.
    """
    
    def __init__(self):
        """
        Initialize the ConnectionManager.
        """
        self.active_connections: Dict[uuid.UUID, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> uuid.UUID:
        """
        Connect a new WebSocket and return a session ID.
        
        Args:
            websocket (WebSocket): The WebSocket to connect.
        
        Returns:
            uuid.UUID: A new session ID for the connection.
        """
        await websocket.accept()
        session_id = uuid.uuid4()
        self.active_connections[session_id] = websocket
        return session_id

    def disconnect(self, session_id: uuid.UUID):
        """
        Disconnect a WebSocket connection.
        
        Args:
            session_id (uuid.UUID): The session ID to disconnect.
        """
        self.active_connections.pop(session_id, None)

    async def send_personal_message(self, message: str, session_id: uuid.UUID):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message (str): The message to send.
            session_id (uuid.UUID): The session ID to send the message to.
        """
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This function handles the startup and shutdown events for the application.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Startup
    global chatbot
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URI"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    # Create OpenAI embedding function
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_EMB_ENDPOINT_URI"),
        api_type="azure",
        api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"),
    )

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="./chromadb_csv", settings=Settings(anonymized_telemetry=False))

    # Initialize RAGChatbot with the created clients
    chatbot = RAGChatbot(client, openai_ef, chroma_client)

    yield

    # Shutdown
    db.disconnect()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Initialize ConnectionManager
manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the chat application.
    
    This function handles incoming WebSocket connections and messages.
    
    Args:
        websocket (WebSocket): The WebSocket connection.
    """
    session_id = await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await chatbot.process_message(websocket, message, session_id)
    except WebSocketDisconnect:
        manager.disconnect(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


import logging
import os
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import asyncio
import colorlog
import chromadb
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from pony.orm import Database, Required, Optional as PonyOptional, Json, Set, db_session, commit
from pony.orm.asynchronous import set_sql_debug

# Load environment variables
load_dotenv('.env')

# Database setup
db = Database()

class ChatSession(db.Entity):
    """
    Represents a chat session in the database.
    
    Attributes:
        id (uuid.UUID): Unique identifier for the chat session.
        created_at (datetime): Timestamp when the session was created.
        updated_at (datetime): Timestamp when the session was last updated.
        chat_history (Json): JSON representation of the chat history.
    """
    id = Required(uuid.UUID, default=uuid.uuid4, primary_key=True)
    created_at = Required(datetime, default=lambda: datetime.now(timezone.utc))
    updated_at = Required(datetime, default=lambda: datetime.now(timezone.utc))
    chat_history = Required(Json)

# Connect to the database
db.bind(provider='postgres', host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), 
        password=os.getenv('DB_PASSWORD'), database=os.getenv('DB_NAME'))
db.generate_mapping(create_tables=True)

set_sql_debug(True)  # Enable SQL debug logging

class RAGChatbot:
    """
    Retrieval-Augmented Generation (RAG) Chatbot class.
    
    This class handles the core functionality of the chatbot, including
    data retrieval, response generation, and message processing.
    """

    def __init__(self, client: AsyncAzureOpenAI, openai_ef: OpenAIEmbeddingFunction, chroma_client: chromadb.PersistentClient):
        """
        Initialize the RAGChatbot.

        Args:
            client (AsyncAzureOpenAI): Azure OpenAI client for generating responses.
            openai_ef (OpenAIEmbeddingFunction): OpenAI embedding function for text embeddings.
            chroma_client (chromadb.PersistentClient): Chroma client for vector database operations.
        """
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set up logger
        self.logger = self._setup_logger()

        # Initialize clients
        self.client = client
        self.openai_ef = openai_ef
        self.chroma_client = chroma_client

        # Load the collections
        self.excel_collection = self.chroma_client.get_collection(name="test", embedding_function=self.openai_ef)
        self.pdf_collection = self.chroma_client.get_collection(name="pdf", embedding_function=self.openai_ef)

    def _setup_logger(self):
        """
        Set up and configure the logger for the chatbot.

        Returns:
            logging.Logger: Configured logger instance.
        """
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

    async def retrieve_excel_data(self, query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant Excel data based on the query.

        Args:
            query (str): The user's query.
            top_k (int): Number of top results to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing relevant Excel data, or None if no data found.
        """
        self.logger.info(f"Retrieving top {top_k} relevant rows for query: '{query}'")
        
        results = self.excel_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        
        if not results['ids'][0]:
            self.logger.info("No relevant Excel data found.")
            return None
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0]
        }

    async def retrieve_pdf_data(self, query: str, top_k: int = 5) -> Optional[list]:
        """
        Retrieve relevant PDF data based on the query.

        Args:
            query (str): The user's query.
            top_k (int): Number of top results to retrieve.

        Returns:
            Optional[list]: List of relevant PDF data, or None if no data found.
        """
        self.logger.info(f"Performing PDF document search for query: {query}")
        results = self.pdf_collection.query(query_texts=[query], n_results=top_k)
        
        if not results["documents"][0]:
            self.logger.info("No relevant PDF data found.")
            return None
        
        return results["documents"][0]

    async def concurrent_retrieval(self, query: str, top_k: int = 5) -> tuple:
        """
        Perform concurrent retrieval of Excel and PDF data.

        Args:
            query (str): The user's query.
            top_k (int): Number of top results to retrieve.

        Returns:
            tuple: A tuple containing Excel and PDF results.
        """
        excel_results, pdf_results = await asyncio.gather(
            self.retrieve_excel_data(query, top_k),
            self.retrieve_pdf_data(query, top_k)
        )
        return excel_results, pdf_results

    def format_excel_data(self, excel_results: Optional[Dict[str, Any]]) -> str:
        """
        Format Excel data for inclusion in the chatbot's response.

        Args:
            excel_results (Optional[Dict[str, Any]]): Excel data to format.

        Returns:
            str: Formatted Excel data as a string.
        """
        if excel_results is None:
            return "No relevant Excel data found."
        
        return "\n".join([f"Row {i+1}: {metadata}" for i, metadata in enumerate(excel_results['metadatas'])])

    def format_pdf_data(self, pdf_results: Optional[list]) -> str:
        """
        Format PDF data for inclusion in the chatbot's response.

        Args:
            pdf_results (Optional[list]): PDF data to format.

        Returns:
            str: Formatted PDF data as a string.
        """
        if pdf_results is None:
            return "No relevant PDF data found."
        
        return "\n".join(pdf_results)

    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format the chat history for inclusion in the chatbot's context.

        Args:
            chat_history (List[Dict[str, str]]): List of chat messages.

        Returns:
            str: Formatted chat history as a JSON string.
        """
        return json.dumps(chat_history, indent=2)

    async def generate_response(self, augmented_input: str) -> str:
        """
        Generate a response using the Azure OpenAI client.

        Args:
            augmented_input (str): The augmented user input including context and retrieved data.

        Returns:
            str: Generated response from the AI model.
        """
        self.logger.info("Generating response using Azure OpenAI...")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided data.",
            },
            {"role": "user", "content": augmented_input},
        ]
        
        chat_response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
        )
        
        return chat_response.choices[0].message.content

    async def rag(self, query: str, chat_history: List[Dict[str, str]], columns: List[str] = None) -> str:
        """
        Perform the Retrieval-Augmented Generation (RAG) process.

        Args:
            query (str): The user's query.
            chat_history (List[Dict[str, str]]): List of previous chat messages.
            columns (List[str], optional): List of column names for Excel data.

        Returns:
            str: Generated response based on the query and retrieved data.
        """
        self.logger.info(f"Starting RAG pipeline for query: '{query}'")
        
        # Get a sample document to extract column names if not provided
        if not columns:
            sample_results = self.excel_collection.peek(limit=1)
            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                columns = list(sample_results['metadatas'][0].keys())
                primary_column = columns[0] 
                print(f"Primary column: {primary_column}")
            else:
                self.logger.error("Unable to retrieve column information from the collection.")
                return "Error: Unable to process the query due to missing column information."

        # Retrieve relevant data concurrently
        excel_results, pdf_results = await self.concurrent_retrieval(query, top_k=5)
        
        # Format the results
        excel_data = self.format_excel_data(excel_results)
        pdf_data = self.format_pdf_data(pdf_results)
        
        # Format chat history as JSON
        formatted_history = self.format_chat_history(chat_history)
        
        # Augment the query
        augmented_input = f"""Chat History (JSON format):
        {formatted_history}

        Current Query: {query}

        Relevant Data In Excel:
        {excel_data}

        Available Columns: {', '.join(columns)}

        Relevant Data In PDF:
        {pdf_data}

        Instructions:
        1. The chat history is provided in JSON format. Each object in the array represents a complete exchange, with 'user' and 'assistant' keys.
        2. Consider the chat history and the current query to provide a contextual response.
        3. Use the provided data to answer the query as accurately as possible.
        4. If no information was found in either Excel or PDF data, explicitly state this in your response.
        5. If information was found in one source but not the other, mention this and use the available information to answer the query.
        6. Provide context and explanations where appropriate.
        7. Use the available columns to structure your response when referring to Excel data.
        8. Maintain a conversational tone, acknowledging previous interactions when relevant.

        Based on the above information and instructions, please answer the current query in the context of the ongoing conversation."""
        
        # Generate response using Azure OpenAI
        response = await self.generate_response(augmented_input)
        
        self.logger.info("RAG pipeline completed successfully.")
        return response

    async def process_message(self, websocket: WebSocket, message: str, session_id: uuid.UUID):
        """
        Process an incoming message from a WebSocket connection.

        Args:
            websocket (WebSocket): The WebSocket connection.
            message (str): The user's message.
            session_id (uuid.UUID): The unique identifier for the chat session.
        """
        try:
            with db_session:
                chat_session = ChatSession.get(id=session_id)
                if not chat_session:
                    chat_session = ChatSession(id=session_id, chat_history=[])
                
                chat_history = chat_session.chat_history

                # Process the message and get the response
                response = await self.rag(message, chat_history)
                
                # Update chat history
                chat_history.append({"user": message, "assistant": response})
                if len(chat_history) > 5:
                    chat_history.pop(0)
                
                chat_session.chat_history = chat_history
                chat_session.updated_at = datetime.now(timezone.utc)
                
                commit()  # Commit the changes to the database
            
            # Send the response back to the client
            await websocket.send_json({"response": response, "chat_history": chat_history})
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            await websocket.send_json({"error": "An error occurred while processing your message."})

class ConnectionManager:
    """
    Manages WebSocket connections for the chat application.
    """

    def __init__(self):
        """
        Initialize the ConnectionManager.
        """
        self.active_connections: Dict[uuid.UUID, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> uuid.UUID:
        """
        Connect a new WebSocket and return a unique session ID.

        Args:
            websocket (WebSocket): The WebSocket connection to add.

        Returns:
            uuid.UUID: A unique session ID for the connection.
        """
        await websocket.accept()
        session_id = uuid.uuid4()
        self.active_connections[session_id] = websocket
        return session_id

    def disconnect(self, session_id: uuid.UUID):
        """
        Disconnect a WebSocket connection.

        Args:
            session_id (uuid.UUID): The session ID of the connection to remove.
        """
        del self.active_connections[session_id]

    async def broadcast(self, message: str):
        """
        Broadcast a message to all active connections.

        Args:
            message (str): The message to broadcast.
        """
        for connection in self.active_connections.values():
            await connection.send_text(message)

class ChatbotApp:
    """
    Main application class for the RAG Chatbot.
    """

    def __init__(self):
        """
        Initialize the ChatbotApp.
        """
        self.app = FastAPI()
        self.chatbot = None
        self.connection_manager = ConnectionManager()
        self.setup_routes()

    def setup_routes(self):
        """
        Set up the FastAPI routes for the application.
        """
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            session_id = await self.connection_manager.connect(websocket)
            try:
                while True:
                    message = await websocket.receive_text()
                    await self.chatbot.process_message(websocket, message, session_id)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(session_id)

        @self.app.on_event("startup")
        async def startup():
            # Initialize Azure OpenAI client
            client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URI"),
                api_key=os.getenv("AZURE_OPEN
