import logging
import requests
from O365 import Account, MSGraphProtocol, FileSystemTokenBackend
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.config import Settings


class Office365Authenticator:
    """Handles authentication and session management for Microsoft Graph API"""

    def __init__(self, config: Settings):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = self._configure_session()
        self.account = self._initialize_account()

    def _configure_session(self) -> requests.Session:
        """Creates a secure HTTP session with retries"""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        session = requests.Session()
        session.verify = True
        session.trust_env = False
        session.mount("https://", adapter)
        return session

    def _initialize_account(self):
        """Initializes Office 365 authentication"""
        protocol = MSGraphProtocol()
        token_backend = FileSystemTokenBackend(token_path="office365_token")

        account = Account(
            credentials=(self.config.CLIENT_ID, self.config.CLIENT_SECRET),
            tenant_id=self.config.TENANT_ID,
            protocol=protocol,
            token_backend=token_backend,
            auth_flow_type="credentials",
            request_session=self.session
        )
        return account

    def authenticate(self) -> bool:
        """Ensures authentication and refreshes token if needed"""
        try:
            success = self.account.authenticate(scopes=["https://graph.microsoft.com/.default"])
            if success:
                auth_token = self.account.con.token_backend.token
                self.session.headers.update({
                    "Authorization": f"Bearer {auth_token['access_token']}",
                    "Content-Type": "application/json"
                })
                return True
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
        return False