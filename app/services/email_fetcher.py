import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from app.services.email_auth import Office365Authenticator


class EmailFetcher:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.authenticator = Office365Authenticator(config)

    def fetch_emails(
            self,
            days_back: int,
            senders: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetch emails matching specified criteria"""
        try:
            if not self.authenticator.authenticate():
                raise RuntimeError("Failed to authenticate with Office 365")

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