from typing import List, Dict

from pydantic import BaseModel


class EmailChunk(BaseModel):
    chunk_id: str
    email_id: str
    content: str
    embedding: List[float]
    chunk_index: int
    metadata: Dict[str, any]