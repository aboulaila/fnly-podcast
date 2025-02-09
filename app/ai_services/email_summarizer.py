import logging

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain


class EmailSummarizer(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.summarizer = self._create_summarizer()

    def extract_headlines(self, text: str) -> str:
        try:
            doc = Document(page_content=text)
            result = self.summarizer.invoke({"input_documents": [doc]})
            return result.get("output_text", "")
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return text[:500] + "..."

    @staticmethod
    def _create_summarizer():
        summarize_prompt = PromptTemplate(
            template="""extract newsletters headlines from the below email body
                            {text}""",
            input_variables=["text"]
        )

        return load_summarize_chain(
            llm=ChatOpenAI(model_name="gpt-4o-mini"),
            chain_type="stuff",
            prompt=summarize_prompt
        )