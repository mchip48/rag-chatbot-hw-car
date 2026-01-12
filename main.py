# def main():
#     print("Hello from rag-chatbot-hw-car!")


# if __name__ == "__main__":
#     main()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

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
conversations = {}

class ChatMessage(BaseModel):
    message: str
    conversation_id: str = "default"

@app.get("/")
def index():
    return {
        "message": "Chatbot API",
        "endpoints": {
            "POST /chat": "Send a message to the chatbot",
            "GET /conversations/{conversation_id}": "Get conversation history",
            "DELETE /conversations/{conversation_id}": "Clear conversation history"
        }
    }

@app.post("/chat")
def create(chat_message: ChatMessage):
    conversation_id = chat_message.conversation_id
    user_message = chat_message.message

    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {"role": "developer", "content": "You are a helpful AI assistant."}
        ]

    conversations[conversation_id].append({"role": "user", "content": user_message})

    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=1,
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
    if conversation_id not in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted!"}
    return {"error": "Conversation not found"}