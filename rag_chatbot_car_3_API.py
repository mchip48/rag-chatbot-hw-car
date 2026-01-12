import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("rag-chatbot-hw-car")

conversations = {}


class ChatMessage(BaseModel):
    message: str
    conversation_id: str = "default"


# ========== Helper Functions ==========

def rag(user_input):
    """Search Pinecone for relevant documentation"""
    results = dense_index.search(
        namespace="car-maintenance",
        query={
            "top_k": 3,
            "inputs": {"text": user_input}
        }
    )
    
    documentation = ""
    for hit in results['result']['hits']:
        fields = hit.get('fields')
        chunk_text = fields.get('chunk_text')
        documentation += chunk_text
    
    return documentation


def system_prompt():
    """Return the system prompt for the chatbot"""
    return {
        "role": "developer",
        "content": """You are an AI customer support technician who is knowledgeable 
        about software products created by the company called GROSS. The products are:
        * Flamehamster, a web browser.
        * Rumblechirp, an email client.
        * GuineaPigment, a drawing tool for creating/editing SVGs
        * EMRgency, an electronic medical record system
        * Verbiage++, a content management system."""
    }


def user_prompt(user_input, documentation):
    """Return the user prompt with RAG context"""
    return {
        "role": "user",
        "content": f"""Here are excerpts from the official GROSS documentation: 
        {documentation}. Use whatever info from the above documentation excerpts 
        (and no other info) to answer the following query: {user_input}. 
        If the user asks something that you are unsure of, make sure to always 
        ask follow-up questions to make sure you're clear on what the user needs. 
        Also, if the user asks something that is vague and you're not sure what 
        service they're asking about, ask follow up questions."""
    }


# ========== API Endpoints ==========

@app.get("/")
def index():
    return {
        "message": "GROSS Support Chatbot API (with RAG)",
        "endpoints": {
            "POST /chat": "Send a message (uses RAG)",
            "GET /conversations/{id}": "Get conversation history",
            "DELETE /conversations/{id}": "Clear conversation"
        }
    }


@app.post("/chat")
def create(chat_message: ChatMessage):
    conversation_id = chat_message.conversation_id
    user_message = chat_message.message
    
    # Initialize conversation if new
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            system_prompt(),
            {"role": "assistant", "content": "How can I help you today?"}
        ]
    
    # Get relevant documentation via RAG
    documentation = rag(user_message)
    
    # Add user message with RAG context
    conversations[conversation_id].append(user_prompt(user_message, documentation))
    
    # Get response from LLM
    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        input=conversations[conversation_id]
    )
    
    assistant_message = response.output_text
    conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
    
    return {
        "message": assistant_message,
        "conversation_id": conversation_id
    }


@app.get("/conversations/{conversation_id}")
def show(conversation_id: str):
    if conversation_id not in conversations:
        return {"error": "Conversation not found"}
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id]
    }


@app.delete("/conversations/{conversation_id}")
def destroy(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted"}
    return {"error": "Conversation not found"}