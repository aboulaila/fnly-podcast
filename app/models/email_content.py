from datetime import datetime
from typing import List, Dict, Any

from pydantic import BaseModel


class EmailContent(BaseModel):
    """Structured representation of processed email content"""
    subject: str
    sender: str
    received_date: datetime
    processed_text: str
    links: List[str]
    metadata: Dict[str, Any]