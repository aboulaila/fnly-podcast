import json
from typing import Any

from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from pydantic import Field

from app.models.analysis_result import AnalysisResult
from app.services.email_analysis_store import EmailAnalysisStore
from app.services.email_vector_store import EmailVectorStore


class AnalysisTool(BaseTool):
    llm: Any = Field(default=None)
    qa_chain: Any = Field(default=None)
    retriever: VectorStoreRetriever = Field(default=None)
    storage_manager: EmailVectorStore = Field(default=None)
    email_analysis_store: EmailAnalysisStore = Field(default=None)

    def __init__(self, llm, vector_store: PineconeVectorStore, storage_manager: EmailVectorStore, email_analysis_store: EmailAnalysisStore):
        super().__init__(name="AnalysisTool", description=(
            "Identify and categorize key topics and themes discussed in the newsletters."
            "Summarize complex information to highlight essential points."
        ))
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        self.storage_manager = storage_manager
        self.email_analysis_store = email_analysis_store
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
            chunks = self.storage_manager.get_email_chunks(email_id, metadata.num_chunks)
            meta_data = json.dumps(metadata.to_dict())
            context_texts = [doc[0].page_content for doc in chunks]
            analysis_result = self.qa_chain.invoke({"knowledge_base_context": context_texts, "metadata": meta_data})

            result = self.email_analysis_store.store_analysis(analysis_result[0])
            return {
                "analysis_id": result.analysis_id,
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