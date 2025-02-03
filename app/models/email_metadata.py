from datetime import datetime

from pydantic import BaseModel


class EmailMetadata(BaseModel):
    email_id: str
    subject: str
    sender: str
    timestamp: datetime
    headlines: str
    content: str
    links: list[str]
    num_chunks: int
    total_tokens: int

    def to_dict(self):
        metadata_dict = self.model_dump(mode="json")

        # Convert datetime to string if present
        if isinstance(metadata_dict.get('timestamp'), datetime):
            metadata_dict['timestamp'] = metadata_dict['timestamp'].isoformat()

        return metadata_dict