from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.schema import SystemMessage, HumanMessagePromptTemplate

# Steps 1 and 2 remain the same (document loading and vector store creation)

# Step 3: Create the prompt template using from_messages
system_message = """You are a personal assistant specialized in recommending car types and models based on detailed personality profiles. Each profile will include a description of the persona, their preferences, and specific features they value in a vehicle. Your task is to suggest suitable Nissan car models that align with the given persona's preferences. Provide more than one option if possible. Each suggestion should include the car type, model, and a reason explaining why it fits the persona's needs. Format the output as a JSON object.

Instructions:
1. Analyze the provided persona's description, keywords, visual and textual contents, points of focus, emotional drivers, demographics, and behavioral traits.
2. Recommend suitable Nissan car models that match the persona's preferences.
3. Ensure each recommendation includes:
   * Car type (e.g., Sedan, SUV, Compact Car, etc.)
   * Car model (specific Nissan model)
   * Reason (explanation why this model is a good fit for the persona)
4. Structure the output as a JSON object with the persona name and an array of car suggestions."""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_message),
    HumanMessagePromptTemplate.from_template("Context: {context}\n\nPersona: {question}")
])

# Step 4: Create the LLM
llm = ChatOpenAI()

# Step 5: Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Step 6: Create the retrieval chain
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 7: Function to process persona and get car suggestions
def get_car_suggestions(persona_data):
    response = retrieval_chain.invoke({"question": str(persona_data)})
    return response["answer"]

# Example usage
persona_data = {
    "persona": "The Brand Conservative/Practical Buyer",
    "description": "Focuses on finding the best value for their money. Prioritizes safety, reliability, and fuel efficiency over brand appeal or style. Likely to research extensively before making a purchase decision.",
    "keywords": ["Quality", "Safety Features", "Fuel Efficiency", "Practicality", "Reliability", "Affordability", "Low Maintenance"],
    "visual_textual_contents": [
        "Clear and concise information about fuel efficiency and safety ratings",
        "Easy-to-use comparison tools",
        "Simple and straightforward navigation",
        "Focus on value and affordability",
        "Images and videos that showcase the car's fuel efficiency, safety features, and practicality",
        "Shots of the car in a city setting or on a long road trip",
        "Videos highlighting the car's handling, braking, and acceleration"
    ],
    "points_of_focus": ["Reviews and comparison", "Safety and Reliability", "Fuel efficiency", "Practical Features"],
    "emotional_drivers": "Make me feel in control",
    "demographics": "Age: 35-55, Income: Middle to High, Occupation: Professional",
    "behavioral_traits": "Thorough researcher, Value-oriented, Risk-averse"
}

suggestions = get_car_suggestions(persona_data)
print(suggestions)
