from dotenv import load_dotenv
load_dotenv()

from app.app import App

from fastapi import FastAPI
from langsmith.middleware import TracingMiddleware
api = FastAPI()
api.add_middleware(TracingMiddleware)
app = App()

app.init()

@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
