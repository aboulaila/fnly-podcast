from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CLIENT_ID: str
    CLIENT_SECRET: str
    TENANT_ID: str
    USER_ID: str
    ANTHROPIC_API_KEY: str
    RECEIVER_EMAIL: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    PINECONE_API_KEY: str
    ASTRA_DB_KEYSPACE: str
    ASTRA_DB_API_ENDPOINT: str
    ASTRA_DB_APPLICATION_TOKEN: str

    model_config = {
        "env_file": ".env",
    }


settings = Settings()