import logging
from datetime import datetime

from app.config import settings
from app.email_client import EmailClient
from app.content_extractor import ContentExtractor
from app.ai_analyzer import NewsletterAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class App:
    def init(self):
        try:
            # Initialize components
            email_client = EmailClient(settings)

            # Authenticate with Office 365
            logger.info("Authenticating with Office 365...")
            if not email_client.authenticate():
                logger.error("Authentication failed")
                return

            content_extractor = ContentExtractor()
            analyzer = NewsletterAnalyzer()

            # Fetch newsletters
            logger.info("Fetching newsletters...")
            newsletters = email_client.fetch_newsletters(3, senders=["news@alphasignal.ai", "info@youreverydayai.com", "dan@tldrnewsletter.com", "noreply@medium.com"])

            if not newsletters:
                logger.info("No newsletters found in the specified time range.")
                return

            # Extract content from newsletters
            logger.info("Extracting content from newsletters...")
            newsletter_contents = []
            for email in newsletters:
                content = content_extractor.extract_content(email)
                if content:
                    newsletter_contents.append(content)

            # Analyze and generate summary
            logger.info("Generating summary...")
            summary = analyzer.analyze_newsletters(newsletter_contents)

            if summary:
                # Format the email content
                email_subject = "AI Newsletter Summary Report"
                email_body = summary

                # Send the summary via email
                logger.info("Sending summary email...")
                if email_client.send_email(email_subject, email_body):
                    logger.info("Summary email sent successfully")
                else:
                    logger.error("Failed to send summary email")
            else:
                logger.error("Failed to generate summary.")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def _format_email_content(self, summary, newsletter_contents):
        """Format the email content with summary and metadata"""
        current_date = datetime.now().strftime("%Y-%m-%d")

        email_content = [
            f"Newsletter Summary Report - {current_date}",
            "=" * 50,
            "\nAnalyzed Newsletters:",
            "-" * 20
        ]

        # Add metadata about analyzed newsletters
        for content in newsletter_contents:
            email_content.append(f"\n• {content['subject']}")
            email_content.append(f"  From: {content['sender']}")

        # Add the AI-generated summary
        email_content.extend([
            "\nExecutive Summary:",
            "-" * 20,
            summary
        ])

        # Add footer
        email_content.extend([
            "\n" + "=" * 50,
            "This summary was generated automatically by the Newsletter Analysis System.",
            "For questions or feedback, please contact your system administrator."
        ])

        return "\n".join(email_content)