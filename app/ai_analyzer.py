from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json


@dataclass
class NewsletterContent:
    subject: str
    sender: str
    text: str
    links: List[str]
    date: Optional[datetime] = None


@dataclass
class ProcessedNewsletter:
    subject: str
    key_points: List[str]
    main_topics: List[str]
    important_links: List[Dict[str, str]]
    sentiment: str
    priority_level: str
    date: Optional[datetime] = None


class NewsletterAnalyzer:
    """A two-stage newsletter analysis system using different models for processing and summarization."""

    def __init__(
            self,
            first_stage_model: str = "gpt-3.5-turbo",  # Lighter model for initial processing
            second_stage_model: str = "claude-3-5-sonnet-20240620",  # More sophisticated model for final summary
            batch_size: int = 3,  # Number of newsletters to process in each batch
            temperature: float = 0.7,
            max_tokens: int = 4000
    ):
        self.first_stage = ChatOpenAI(
            model_name=first_stage_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.second_stage = ChatAnthropic(
            model=second_stage_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def analyze_newsletters(
            self,
            newsletters_content: List[Dict[str, Union[str, List[str]]]]
    ) -> Optional[str]:
        """Main method to process and analyze newsletters in two stages."""
        try:
            self.logger.info(f"Starting analysis of {len(newsletters_content)} newsletters")

            # Stage 1: Process newsletters in batches
            processed_newsletters = []
            for i in range(0, len(newsletters_content), self.batch_size):
                batch = newsletters_content[i:i + self.batch_size]
                processed_batch = self._process_newsletter_batch(batch)
                processed_newsletters.extend(processed_batch)

            # Stage 2: Generate final summary
            if processed_newsletters:
                return self._generate_final_summary(processed_newsletters)
            return None

        except Exception as e:
            self.logger.error(f"Error in newsletter analysis: {str(e)}", exc_info=True)
            return None

    def _process_newsletter_batch(
            self,
            batch: List[Dict[str, Union[str, List[str]]]]
    ) -> List[ProcessedNewsletter]:
        """First stage: Process a batch of newsletters using the lighter model."""

        first_stage_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an efficient newsletter analyzer. Your task is to extract key information from each newsletter and provide a structured analysis. For each newsletter, provide:

1. A list of 3-5 key points
2. Main topics (maximum 3)
3. Important links with context
4. Overall sentiment (positive/neutral/negative)
5. Priority level (high/medium/low)

Return the analysis in JSON format with these exact fields:
{{
    "key_points": [],
    "main_topics": [],
    "important_links": [{{"url": "", "context": ""}}],
    "sentiment": "",
    "priority_level": ""
}}"""
            ),
            ("human", "{input}")
        ])

        processed_results = []
        for newsletter in batch:
            try:
                formatted_content = self._prepare_single_newsletter(newsletter)
                response = self.first_stage.invoke(
                    first_stage_prompt.invoke({"input": formatted_content})
                )

                analysis = json.loads(response.content)
                processed_results.append(
                    ProcessedNewsletter(
                        subject=newsletter.get('subject', ''),
                        key_points=analysis['key_points'],
                        main_topics=analysis['main_topics'],
                        important_links=analysis['important_links'],
                        sentiment=analysis['sentiment'],
                        priority_level=analysis['priority_level'],
                        date=self._parse_date(newsletter.get('date'))
                    )
                )
            except Exception as e:
                self.logger.error(f"Error processing newsletter: {str(e)}")
                continue

        return processed_results

    def _generate_final_summary(self, processed_newsletters: List[ProcessedNewsletter]) -> str:
        """Second stage: Generate final HTML summary using the sophisticated model."""

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Create a comprehensive HTML summary of the analyzed newsletters. The summary should:

1. Group content by main topics and priority levels
2. Highlight key insights and trends
3. Include relevant links with context
4. Use semantic HTML5 with Apple-style formatting
5. Be organized for ~15 minute reading time

Focus on creating a cohesive narrative that connects related points across newsletters.
Return only the HTML content."""
            ),
            ("human", "{input}")
        ])

        # Prepare processed content for final summary
        summary_input = self._prepare_processed_content(processed_newsletters)
        response = self.second_stage.invoke(
            summary_prompt.invoke({"input": summary_input})
        )

        return self._extract_html_content(response.content)

    def _prepare_single_newsletter(self, newsletter: Dict) -> str:
        """Format a single newsletter for first-stage analysis."""
        return (
            f"Subject: {newsletter.get('subject', '')}\n"
            f"Content: {newsletter.get('text', '')}\n"
            f"Links: {', '.join(newsletter.get('links', []))}"
        )

    def _prepare_processed_content(self, newsletters: List[ProcessedNewsletter]) -> str:
        """Format processed newsletters for second-stage summary."""
        formatted_content = []
        for newsletter in newsletters:
            formatted_content.append(
                f"Subject: {newsletter.subject}\n"
                f"Key Points: {json.dumps(newsletter.key_points)}\n"
                f"Main Topics: {', '.join(newsletter.main_topics)}\n"
                f"Priority: {newsletter.priority_level}\n"
                f"Sentiment: {newsletter.sentiment}\n"
                f"Important Links: {json.dumps(newsletter.important_links)}\n"
                "---"
            )
        return "\n".join(formatted_content)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            self.logger.warning(f"Invalid date format: {date_str}")
            return None

    def _extract_html_content(self, content: str) -> str:
        """Extract HTML content from the response."""
        start_idx = content.find('<html')
        end_idx = content.rfind('>') + 1

        if start_idx == -1 or end_idx == -1:
            self.logger.warning("HTML tags not found in response")
            return content

        return content[start_idx:end_idx]