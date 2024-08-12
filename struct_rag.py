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
from datetime import datetime
from pony.orm import Database, Required, Optional, Json, Set
from pony.orm.asynchronous import set_sql_debug

# Load environment variables
load_dotenv('.env')

# Database setup
db = Database()

class ChatSession(db.Entity):
    id = Required(uuid.UUID, default=uuid.uuid4, primary_key=True)
    created_at = Required(datetime, default=datetime.utcnow)
    updated_at = Required(datetime, default=datetime.utcnow)
    chat_history = Required(Json)

# Connect to the database
db.bind(provider='postgres', host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), 
        password=os.getenv('DB_PASSWORD'), database=os.getenv('DB_NAME'))
db.generate_mapping(create_tables=True)

set_sql_debug(True)  # Enable SQL debug logging

class RAGChatbot:
    def __init__(self):
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set up logger
        self.logger = self.setup_logger()

        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URI"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        # Create OpenAI embedding function
        self.openai_ef = OpenAIEmbeddingFunction(
            api_key=os.getenv("AZURE_OPENAI_EMB_API_KEY"),
            api_base=os.getenv("AZURE_OPENAI_EMB_ENDPOINT_URI"),
            api_type="azure",
            api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION"),
        )

        # Initialize Chroma client and load the collections
        self.chroma_client = chromadb.PersistentClient(path="./chromadb_csv", settings=Settings(anonymized_telemetry=False))
        self.excel_collection = self.chroma_client.get_collection(name="test", embedding_function=self.openai_ef)
        self.pdf_collection = self.chroma_client.get_collection(name="pdf", embedding_function=self.openai_ef)

    def setup_logger(self):
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
        self.logger.info(f"Performing PDF document search for query: {query}")
        results = self.pdf_collection.query(query_texts=[query], n_results=top_k)
        
        if not results["documents"][0]:
            self.logger.info("No relevant PDF data found.")
            return None
        
        return results["documents"][0]

    async def concurrent_retrieval(self, query: str, top_k: int = 5) -> tuple:
        excel_results, pdf_results = await asyncio.gather(
            self.retrieve_excel_data(query, top_k),
            self.retrieve_pdf_data(query, top_k)
        )
        return excel_results, pdf_results

    def format_excel_data(self, excel_results: Optional[Dict[str, Any]]) -> str:
        if excel_results is None:
            return "No relevant Excel data found."
        
        return "\n".join([f"Row {i+1}: {metadata}" for i, metadata in enumerate(excel_results['metadatas'])])

    def format_pdf_data(self, pdf_results: Optional[list]) -> str:
        if pdf_results is None:
            return "No relevant PDF data found."
        
        return "\n".join(pdf_results)

    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        return json.dumps(chat_history, indent=2)

    async def generate_response(self, augmented_input: str) -> str:
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
        try:
            @db.transaction(retry=3)
            async def process_and_save():
                chat_session = await ChatSession.get(id=session_id)
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
                chat_session.updated_at = datetime.utcnow()
                
                await chat_session.flush()
                return response, chat_history

            response, chat_history = await process_and_save()
            
            # Send the response back to the client
            await websocket.send_json({"response": response, "chat_history": chat_history})
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            await websocket.send_json({"error": "An error occurred while processing your message."})

app = FastAPI()
chatbot = RAGChatbot()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[uuid.UUID, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> uuid.UUID:
        await websocket.accept()
        session_id = uuid.uuid4()
        self.active_connections[session_id] = websocket
        return session_id

    def disconnect(self, session_id: uuid.UUID):
        del self.active_connections[session_id]

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await chatbot.process_message(websocket, message, session_id)
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.on_event("startup")
async def startup():
    # Pony ORM automatically creates tables if they don't exist
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
