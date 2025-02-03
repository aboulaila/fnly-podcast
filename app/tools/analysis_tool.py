import json
from typing import Any, List

from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from pydantic import Field, BaseModel

from app.managers.email_storage_manager import EmailStorageManager


class HeadlineAnalysis(BaseModel):
    """singled headline analysis model."""
    theme: str = Field(description="Theme or topic of the headline")
    key_points: List[str] = Field(description="Key insights and implications of each topic.")
    summary: List[str] = Field(description="Summary of the provided topic")
    relevant_links: List[str] = Field(description="Links or references relevant to the analysis.")
    priority_level: float = Field(
        description="Priority level of the headline. Higher values indicate higher priority.")

class AnalysisResult(BaseModel):
    """Structured analysis result model."""
    headline_insights: List[HeadlineAnalysis] = Field(description="Key insights and implications of each headline.")

class AnalysisTool(BaseTool):
    llm: Any = Field(default=None)
    qa_chain: Any = Field(default=None)
    retriever: VectorStoreRetriever = Field(default=None)
    storage_manager: EmailStorageManager = Field(default=None)

    def __init__(self, llm, vector_store: PineconeVectorStore, storage_manager: EmailStorageManager):
        super().__init__(name="AnalysisTool", description=(
            "Perform sentiment analysis to gauge the emotional tone of the content"
            "Identify and categorize key topics and themes discussed in the newsletters."
            "Extract entities such as names, organizations, and locations mentioned. Summarize complex information to highlight essential points."
        ))
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        self.storage_manager = storage_manager
        self.llm = llm
        self.qa_chain = self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        prompt = self._create_prompt()
        llm_with_tools = self.llm.bind_tools([AnalysisResult])
        output_parser = PydanticToolsParser(tools=[AnalysisResult])
        return (
            prompt
            | llm_with_tools
            | output_parser
        )

    def _run(self, email_id: str):
        try:
            metadata = self.storage_manager.metadata_store.get_metadata(email_id)
            first_chunk = self.storage_manager.get_email_chunks(email_id)
            meta_data = json.dumps({
                "metadata": metadata.to_dict(),
                "preview": first_chunk[0][0].page_content if first_chunk else None
            })
            retrieved_docs = self.retriever.invoke(meta_data)
            context_texts = [doc.page_content for doc in retrieved_docs]
            analysis_result = self.qa_chain.invoke({"knowledge_base_context": context_texts, "metadata": meta_data})
            return {
                "analysis_result": analysis_result[0] if isinstance(analysis_result, list) else analysis_result,
            }
        except Exception as e:
            return {
                "error": str(e)
            }

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports sync.")

    @staticmethod
    def _create_prompt():
        prompt = PromptTemplate(
            input_variables=["metadata", "knowledge_base_context"],
            template="""
                    Analyze the following newsletter content:
                    emails metadata: {metadata}

                    Relevant Knowledge Base Context:
                    {knowledge_base_context}
                    
                    For each headline in the email, provide a detailed analysis
                    """
        )
        return prompt