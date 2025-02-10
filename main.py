from dotenv import load_dotenv

load_dotenv()

from app.agents.orchestrator_agent import OrchestratorAgent

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langsmith.middleware import TracingMiddleware
from pydantic import BaseModel

api = FastAPI()
api.add_middleware(TracingMiddleware)
api.mount("/static", StaticFiles(directory="static"), name="static")

app = OrchestratorAgent()

class ChatMessage(BaseModel):
    message: str

@api.get("/")
async def root():
    return {"message": "Hello World"}

@api.post("/chat")
async def chat(message: ChatMessage):
    """
    Chat interface endpoint that processes messages through the orchestrator agent.
    
    Args:
        message: The chat message/query from the user
    Returns:
        The agent's response
    """
    try:
        response = app.run(message.message)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}, 500


@api.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
