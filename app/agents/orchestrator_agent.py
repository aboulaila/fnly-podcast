from typing import List

from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from pinecone import Pinecone

from app.agents.plan_execution_agent import PlanExecuteAgent
from app.config import settings
from app.services.email_analysis_store import EmailAnalysisStore
from app.services.email_metadata_store import EmailMetadataStore
from app.services.email_vector_store import EmailVectorStore
from app.tools.analysis_tool import AnalysisTool
from app.tools.email_processing_tool import EmailProcessingTool
from app.tools.email_sending_tool import EmailSendingTool
from app.tools.synthesizer_tool import SynthesisTool


class OrchestratorAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
            chunk_size=1000,
        )
        pc = Pinecone()
        index = pc.Index("fnly-podcast")
        self.vector_store = PineconeVectorStore(index, embedding=self.embeddings)

        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.email_storage_manager = EmailVectorStore(vector_store=self.vector_store, metadata_store=EmailMetadataStore())
        self.email_analysis_store = EmailAnalysisStore()

        self.memory = MemorySaver()
        self.tools = self._initialize_tools()

        self.agent = PlanExecuteAgent(
            tools=self.tools
        )

    def _initialize_tools(self) -> List[BaseTool]:
        return [
            AnalysisTool(llm=self.llm, vector_store=self.vector_store, storage_manager=self.email_storage_manager, email_analysis_store=self.email_analysis_store),
            SynthesisTool(llm=self.llm, vector_store=self.vector_store, memory=self.memory, email_analysis_store=self.email_analysis_store),
            EmailProcessingTool(config=settings, email_storage_manager=self.email_storage_manager),
            EmailSendingTool(config=settings)
        ]

    def run(self, user_input: str):
        """
        Process user input as a chat interaction and execute appropriate actions.
        
        Args:
            user_input: The message/query from the user
        """
        result = self.agent.run(user_input)
        return result