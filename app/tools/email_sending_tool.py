import logging
from langchain_core.tools import BaseTool
from pydantic import Field

from app.config import Settings
from app.helpers.email_auth import Office365Authenticator


class EmailSendingTool(BaseTool):
    """Handles sending emails via Microsoft Graph API"""
    config: Settings = Field(default=None)
    logger: logging.Logger = Field(default=None)
    authenticator: Office365Authenticator = Field(default=None)

    def __init__(self, config: Settings):
        super().__init__(
            name="EmailSendingTool",
            description="Sends emails using Microsoft Graph API."
        )
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.authenticator = Office365Authenticator(config)

    def _run(self, subject: str, body: str, recipient: str):
        """Sends an email"""
        if not self.authenticator.authenticate():
            return "Authentication failed."

        endpoint = f"https://graph.microsoft.com/v1.0/users/{self.config.USER_ID}/sendmail"

        email_body = {
            "message": {
                "subject": subject,
                "body": {"contentType": "HTML", "content": body},
                "toRecipients": [{"emailAddress": {"address": recipient}}]
            }
        }

        response = self.authenticator.session.post(endpoint, json=email_body)
        response.raise_for_status()

        logging.info("Email sent successfully")
        return "Successfully sent email."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports sync.")