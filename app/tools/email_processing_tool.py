import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from app.config import Settings
from app.helpers.email_auth import Office365Authenticator
from app.managers.email_processing_manager import EmailProcessingManager
from app.managers.email_storage_manager import EmailStorageManager


class EmailProcessingTool(BaseTool):
    """Fetches emails and extracts relevant newsletter content"""
    config: Settings = Field(default=None)
    logger: logging.Logger = Field(default=None)
    authenticator: Office365Authenticator = Field(default=None)
    content_processing_manager: EmailProcessingManager = Field(default=None)
    email_storage_manager: EmailStorageManager = Field(default=None)

    def __init__(self, config: Settings, email_storage_manager: EmailStorageManager):
        super().__init__(
            name="EmailProcessingTool",
            description=(
                "Fetches emails and extracts newsletter content."
                "Gets metadata about emails and stores them in vector DB"
            )
        )
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.authenticator = Office365Authenticator(config)
        self.content_processing_manager = EmailProcessingManager()
        self.email_storage_manager = email_storage_manager

    def _run(self, days_back: int, senders: Optional[List[str]] = None):
        """Main method to fetch and process emails"""
        try:
            if not self.authenticator.authenticate():
                raise RuntimeError("Failed to authenticate with Office 365")

            raw_emails = self._fetch_emails(days_back, senders)
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

            for email_summary in emails_summaries:
                email_summary.links = []
            
            return {
                "email_summaries": emails_summaries,
            }

        except Exception as e:
            self.logger.error(f"Error in email processing: {str(e)}")
            return {
                "error": str(e)
            }

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports sync.")

    def _fetch_emails(
            self,
            days_back: int,
            senders: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetch emails matching specified criteria"""
        try:
            filter_query = self._build_filter_query(days_back, senders)

            endpoint = (f'https://graph.microsoft.com/v1.0/users/'
                        f'{self.config.USER_ID}/messages')

            params = {
                '$filter': filter_query,
                '$top': 50,
                '$select': 'subject,sender,receivedDateTime,body,bodyPreview',
                '$orderby': 'receivedDateTime desc'
            }

            response = self.authenticator.session.get(endpoint, params=params)
            response.raise_for_status()

            emails = response.json().get('value', [])

            return emails

        except Exception as e:
            self.logger.error(f"Error fetching emails: {str(e)}")
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"Response content: {e.response.content}")
            raise

    @staticmethod
    def _build_filter_query(days_back: int, senders: Optional[List[str]] = None) -> str:
        date_filter = datetime.now() - timedelta(days=days_back)
        date_str = date_filter.strftime('%Y-%m-%d')

        filters = [f"receivedDateTime ge {date_str}"]

        if senders:
            sender_conditions = [f"from/emailAddress/address eq '{sender}'" for sender in senders]
            if sender_conditions:
                filters.append(f"({' or '.join(sender_conditions)})")

        return ' and '.join(filters)