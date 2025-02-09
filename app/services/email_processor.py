import logging
from datetime import datetime
from typing import Dict, Optional, List

from bs4 import BeautifulSoup

from app.models.email_content import EmailContent
from app.services.url_shortener import UrlShortener


class EmailProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.url_shortener = UrlShortener()
    
    def process_email(self, email: Dict) -> Optional[EmailContent]:
        """Process a single email into structured content"""
        try:
            if not email:
                return None

            raw_body = email.get('body', {}).get('content', '')
            processed_text, links = self._extract_content(raw_body)

            cleaned_urls = self.url_shortener.process_urls(links)
            received_date = self._parse_received_date(email.get('receivedDateTime', ''))

            email = EmailContent(
                subject=email.get('subject', ''),
                sender=email.get('sender', {}).get('emailAddress', {}).get('address', ''),
                received_date=received_date,
                processed_text=processed_text,
                links=cleaned_urls,
                metadata={
                    'preview': email.get('bodyPreview', ''),
                    'processed_at': datetime.now()
                }
            )

            if self._validate_email(email):
                return email
            return None

        except Exception as e:
            self.logger.error(f"Error processing email content: {str(e)}")
            return None

    @staticmethod
    def _extract_content(html_content: str) -> tuple[str, List[str]]:
        """Extract clean text and links from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')

        for element in soup(['script', 'style']):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        links = [a.get('href') for a in soup.find_all('a', href=True)]

        return text, links

    @staticmethod
    def _parse_received_date(received_date_str: str) -> datetime:
        if received_date_str:
            if received_date_str.endswith('Z'):
                received_date_str = received_date_str[:-1] + '+00:00'
            return datetime.fromisoformat(received_date_str)
        return datetime.now()

    @staticmethod
    def _validate_email(email_content: EmailContent) -> bool:
        """Validate email content meets minimum requirements"""
        return bool(
            email_content.subject and
            email_content.sender and
            email_content.processed_text
        )