import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.agents.orchestrator_agent import OrchestratorAgent

from fastapi import FastAPI
from langsmith.middleware import TracingMiddleware
api = FastAPI()
api.add_middleware(TracingMiddleware)

app = OrchestratorAgent()
app.run()

@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
