import logging
import os
from datetime import datetime, timedelta

import requests
from O365 import Account, MSGraphProtocol, FileSystemTokenBackend
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Settings


class EmailClient:
    def __init__(self, credentials: Settings):
        self.credentials = (credentials.CLIENT_ID, credentials.CLIENT_SECRET)
        self.tenant_id = credentials.TENANT_ID
        self.user_id = credentials.USER_ID
        self.receiver_email = credentials.RECEIVER_EMAIL
        self.token_path = 'office365_token'

        # Initialize core components
        self._initialize_protocol()
        self._initialize_token_backend()
        self._configure_secure_transport()

        # Initialize account as None
        self.account = None

    def _initialize_protocol(self):
        """Initialize the Microsoft Graph protocol with proper configuration"""
        self.protocol = MSGraphProtocol()
        self.protocol.users_endpoint = '/users'
        self.protocol.protocol_url = 'https://graph.microsoft.com/v1.0'

    def _initialize_token_backend(self):
        """Initialize the token storage system"""
        self.token_backend = FileSystemTokenBackend(token_path=self.token_path)

    def _configure_secure_transport(self):
        """Configure secure transport with retry logic"""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_maxsize=100,
            pool_connections=100
        )

        self.session = requests.Session()
        self.session.verify = True
        self.session.trust_env = False
        self.session.mount('https://', adapter)

    def _ensure_valid_token(self):
        """Ensure we have a valid authentication token"""
        if not self.account or not self.account.is_authenticated:
            return self.authenticate()
        return True

    def authenticate(self):
        """Authenticate with Microsoft Graph API"""
        try:
            os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'

            scopes = ['https://graph.microsoft.com/.default']

            self.account = Account(
                credentials=self.credentials,
                tenant_id=self.tenant_id,
                protocol=self.protocol,
                token_backend=self.token_backend,
                auth_flow_type='credentials',
                request_session=self.session
            )

            auth_success = self.account.authenticate(scopes=scopes)

            if auth_success:
                # Update session headers with new token
                auth_token = self.account.con.token_backend.token
                if auth_token:
                    self.session.headers.update({
                        'Authorization': f'Bearer {auth_token["access_token"]}',
                        'Content-Type': 'application/json'
                    })
                logging.info("Authentication successful")
                return True

            logging.error("Authentication failed")
            return False

        except (ConnectionError, ValueError) as e:
            logging.error(f"Authentication error: {str(e)}")
            return False

    def fetch_newsletters(self, days_back=1, senders=None, newsletter_keywords=None):
        """Fetch newsletter emails with enhanced token management"""
        try:
            if not self._ensure_valid_token():
                raise RuntimeError("Unable to establish valid authentication")

            date_filter = datetime.now() - timedelta(days=days_back)
            date_str = date_filter.strftime("%Y-%m-%d")

            keyword_filters = [f"contains(subject,'{keyword}')" for keyword in newsletter_keywords] \
                if newsletter_keywords else []
            subject_filter = f" and ({' or '.join(keyword_filters)})" if keyword_filters else ""

            senders_filters = [f"from/emailAddress/address eq '{sender}'" for sender in senders] if senders else []
            sender_filter = f" and ({' or '.join(senders_filters)})" if senders else ""

            query_filter = f"receivedDateTime ge {date_str}{sender_filter}{subject_filter}"

            endpoint = f'https://graph.microsoft.com/v1.0/users/{self.user_id}/messages'

            params = {
                '$filter': query_filter,
                '$top': 50,
                '$select': 'subject,sender,receivedDateTime,body,bodyPreview',
                '$orderby': 'receivedDateTime desc'
            }

            response = self.session.get(endpoint, params=params)

            if response.status_code == 401:
                # Token might be expired, attempt refresh and retry
                if self.authenticate():
                    response = self.session.get(endpoint, params=params)

            response.raise_for_status()

            data = response.json()
            newsletters = data.get('value', [])

            if not newsletters:
                logging.info("No newsletters found in the specified time range")

            processed_newsletters = [self.extract_content(newsletter) for newsletter in newsletters]
            return [newsletter for newsletter in processed_newsletters if newsletter]

        except (ConnectionError, ValueError) as e:
            logging.error(f"Error fetching newsletters: {str(e)}")
            self._handle_api_error(e)
            return []

    def extract_content(self, message):
        """Extract content from messages"""
        try:
            if not message:
                return None

            content = {
                'subject': message.get('subject', ''),
                'sender': message.get('sender', {}).get('emailAddress', {}).get('address', ''),
                'received_date': message.get('receivedDateTime', ''),
                'body': message.get('body', {}).get('content', ''),
                'preview': message.get('bodyPreview', '')
            }

            if not content['subject'] or not content['sender']:
                logging.warning("Missing required fields in message")
                return None

            return content

        except Exception as e:
            logging.error(f"Error extracting content: {str(e)}")
            return None

    def send_email(self, subject, body, recipient=None):
        """Send an email with the provided content"""
        try:
            if not self._ensure_valid_token():
                raise RuntimeError("Client not authenticated")

            if recipient is None:
                recipient = self.receiver_email

            endpoint = f'https://graph.microsoft.com/v1.0/users/{self.user_id}/sendmail'

            email_body = {
                "message": {
                    "subject": subject,
                    "body": {
                        "contentType": "HTML",
                        "content": body
                    },
                    "toRecipients": [
                        {
                            "emailAddress": {
                                "address": recipient
                            }
                        }
                    ]
                }
            }

            response = self.session.post(endpoint, json=email_body)
            response.raise_for_status()

            logging.info("Summary email sent successfully")
            return True

        except (ConnectionError, ValueError) as e:
            self._handle_api_error(e)
            return False

    def _handle_api_error(self, error):
        """Handle API errors with token refresh logic"""
        if hasattr(error, 'response') and error.response.content:
            logging.error(f"API error response: {error.response.content}")
            status_code = error.response.status_code
            if status_code == 429:
                logging.error("Rate limit exceeded")
            elif status_code in (401, 403):
                logging.error("Authentication error. Token may need refresh.")
                self.authenticate()
            elif status_code >= 500:
                logging.error("Microsoft Graph API server error.")