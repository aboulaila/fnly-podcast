import logging
from bs4 import BeautifulSoup
import re


class ContentExtractor:
    @staticmethod
    def extract_content(email_message):
        """Extract relevant content from email body"""
        try:
            body = email_message["body"]
            soup = BeautifulSoup(body, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text and links
            text = soup.get_text(separator=' ', strip=True)
            links = [a.get('href') for a in soup.find_all('a', href=True)]

            # Clean up text
            text = re.sub(r'\s+', ' ', text)

            return {
                'subject': email_message["subject"],
                'sender': str(email_message["sender"]),
                'text': text,
                'links': links
            }
        except Exception as e:
            logging.error(f"Error extracting content: {str(e)}")
            return None
