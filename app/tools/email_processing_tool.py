import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from app.config import Settings
from app.services.email_fetcher import EmailFetcher
from app.services.email_processor import EmailProcessor
from app.services.email_vector_store import EmailVectorStore


class EmailProcessingTool(BaseTool):
    """Fetches emails and extracts relevant newsletter content"""
    config: Settings = Field(default=None)
    logger: logging.Logger = Field(default=None)
    email_fetcher: EmailFetcher = Field(default=None)
    content_processing_manager: EmailProcessor = Field(default=None)
    email_storage_manager: EmailVectorStore = Field(default=None)

    def __init__(self, config: Settings, email_storage_manager: EmailVectorStore):
        super().__init__(
            name="EmailProcessingTool",
            description=(
                "Fetches emails and extracts newsletter content."
                "Gets metadata about emails and stores them in vector DB"
            )
        )
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.email_fetcher = EmailFetcher(config)
        self.content_processing_manager = EmailProcessor()
        self.email_storage_manager = email_storage_manager

    def _run(self, days_back: int, senders: Optional[List[str]] = None):
        """Main method to fetch and process emails"""
        try:
            raw_emails = self.email_fetcher.fetch_emails(days_back, senders)
            emails = []
            emails_summaries = []

            for email in raw_emails:
                email = self.content_processing_manager.process_email(email)
                email_metadata = self.email_storage_manager.process_and_store_email(email)
                if email and email_metadata:
                    emails.append(email)
                    emails_summaries.append(email_metadata)
                else:
                    self.logger.warning(f"Email validation failed for subject: {email.get('subject', 'Unknown')}")

            email_ids = [email_summary.email_id for email_summary in emails_summaries]
            
            return {
                "email_ids": email_ids,
            }

        except Exception as e:
            self.logger.error(f"Error in email processing: {str(e)}")
            return {
                "error": str(e)
            }

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports sync.")