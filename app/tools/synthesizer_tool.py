import json
from typing import Any, List

from langchain import hub
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from app.tools.analysis_tool import AnalysisResult


class SynthesisTool(BaseTool):
    llm: Any = Field(default=None)
    llm_chain: Any = Field(default=None)
    retriever: VectorStoreRetriever = Field(default=None)
    memory: MemorySaver = Field(default=None)

    def __init__(self, llm, vector_store: PineconeVectorStore, memory: MemorySaver):
        super().__init__(name="SynthesisTool", description=(
            "Aggregate and organize insights from various analyses into a unified summary."
            "Highlight the most significant findings and actionable items."
            "Format the summary in a user-friendly manner for reporting or sharing purposes."
        ))
        self.llm = llm
        self.llm_chain = self._initialize_llm_chain()
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        self.memory = memory

    def _initialize_llm_chain(self):
        prompt = hub.pull("sythesizer_tool")

        return prompt | self.llm

    def _run(self, analysis_results: List[AnalysisResult]):
        try:
            analysis_results_json = json.dumps([json.dumps(result) for result in analysis_results])
            context = self._retrieve_knowledge_base_context(analysis_results)

            summary = self.llm_chain.invoke(
                {"analysis_result": analysis_results_json, "knowledge_base_context": context}
            )
            return summary
        except Exception as e:

            return {
                "error": str(e)
            }

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool only supports async.")

    def _retrieve_knowledge_base_context(self, analysis_results: List[AnalysisResult]):
        context_documents = []
        for analysis_result in analysis_results:
            for headline in analysis_result["headline_insights"]:
               theme = headline["theme"]
               topics = headline["key_points"]

               topics_str = " ".join(topics)
               query = f"{theme}\n{topics_str}"

               context_documents.extend(self.retriever.invoke(query))

        context = "\n".join([doc.page_content for doc in context_documents])
        return context