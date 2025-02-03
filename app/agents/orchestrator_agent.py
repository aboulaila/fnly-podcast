from datetime import datetime
from typing import List

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from pinecone import Pinecone

from app.agents.react_agent import ReActAgent
from app.config import settings
from app.managers.email_metadata_store import EmailMetadataStore
from app.managers.email_storage_manager import EmailStorageManager
from app.tools.analysis_tool import AnalysisTool
from app.tools.email_processing_tool import EmailProcessingTool
from app.tools.email_sending_tool import EmailSendingTool
from app.tools.synthesizer_tool import SynthesisTool


class OrchestratorAgent:
    # set_llm_cache(AstraDBCache(api_endpoint=settings.ASTRA_DB_API_ENDPOINT, token=settings.ASTRA_DB_APPLICATION_TOKEN, namespace=settings.ASTRA_DB_KEYSPACE))

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
        self.email_storage_manager = EmailStorageManager(llm=self.llm, vector_store=self.vector_store, metadata_store=EmailMetadataStore())

        self.memory = MemorySaver()
        self.tools = self._initialize_tools()
        self.prompt = hub.pull("newsletter_summary")
        self.agent = ReActAgent(self.llm, tools=self.tools, prompt=self.prompt, memory=self.memory)

    def _initialize_tools(self) -> List[BaseTool]:
        return [
            AnalysisTool(llm=self.llm, vector_store=self.vector_store, storage_manager=self.email_storage_manager),
            SynthesisTool(llm=self.llm, vector_store=self.vector_store, memory=self.memory),
            EmailProcessingTool(config=settings, email_storage_manager=self.email_storage_manager),
            EmailSendingTool(config=settings),
            # GetEmailContentTool(storage_manager=self.email_storage_manager)
        ]

    def run(self):
        query = (
            "Generate a newsletter received today from the following recipients: news@alphasignal.ai, info@youreverydayai.com, dan@tldrnewsletter.com, noreply@medium.com"
            f"and then send it to the following email address: {settings.RECEIVER_EMAIL}"
        )
        config = {"configurable": {"thread_id": datetime.now().strftime("%Y%m%d%H%M%S")}}
        result = self.agent.invoke(query, config)
        return result