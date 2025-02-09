import logging
from typing import List, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from app.config import Settings
from app.services.email_fetcher import EmailFetcher
from app.services.email_processor import EmailProcessor


class FetchEmailTool(BaseTool):
    """Fetches emails and extracts relevant newsletter content"""
    config: Settings = Field(default=None)
    logger: logging.Logger = Field(default=None)
    email_fetcher: EmailFetcher = Field(default=None)
    content_processing_manager: EmailProcessor = Field(default=None)

    def __init__(self, config: Settings):
        super().__init__(
            name="FetchEmailTool",
            description=(
                "Fetches email from the connected inbox"
                "Emails can be filtered by date received and sender"
            )
        )
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.email_fetcher = EmailFetcher(config)
        self.content_processing_manager = EmailProcessor()

    def _run(self, days_back: int, senders: Optional[List[str]] = None):
        """Main method to fetch and process emails"""
        try:
            raw_emails = self.email_fetcher.fetch_emails(days_back, senders)
            emails = []

            for email in raw_emails:
                email = self.content_processing_manager.process_email(email)
                if email:
                    emails.append(email)
                else:
                    self.logger.warning(f"Email validation failed for subject: {email.get('subject', 'Unknown')}")

            return {
                "emails": emails,
            }

        except Exception as e:
            self.logger.error(f"Error in email processing: {str(e)}")
            return {
                "error": str(e)
            }

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports sync.")