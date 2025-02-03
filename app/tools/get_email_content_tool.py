from typing import Optional, List

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import Field

from app.managers.email_storage_manager import EmailStorageManager


class GetEmailContentTool(BaseTool):
    storage_manager: EmailStorageManager = Field(default=None)

    def __init__(self, storage_manager: EmailStorageManager):
        super().__init__(
            name="GetEmailContentTool",
            description="Retrieves relevant content using metadata or semantic search"
        )
        self.storage_manager = storage_manager

    def _run(self, email_id: str, query: Optional[str] = None):
        try:
            if query:
                relevant_chunks = self.storage_manager.vector_store.similarity_search(
                    query=query,
                    filter={"email_id": email_id},
                    k=3
                )
                return self._format_chunks(relevant_chunks)
            else:
                metadata = self.storage_manager.metadata_store.get_metadata(email_id)
                first_chunk = self.get_email_chunks(email_id)
                return {
                    "metadata": metadata,
                    "preview": first_chunk[0][0].page_content if first_chunk else None
                }
        except Exception as e:
            return {
                "error": str(e)
            }

    def search_similar_chunks(self, query: str, email_id: str, k: int = 5):
        """Search for similar chunks in Pinecone"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter={"email_id": email_id}
            )
            return results
        except Exception as e:
            print(f"Error searching in Pinecone: {str(e)}")
            return []

    def get_email_chunks(self, email_id: str):
        """Retrieve all chunks for a specific email"""
        try:
            results = self.storage_manager.vector_store.similarity_search_with_score(
                query="",
                filter={"email_id": email_id},
                k=100
            )
            # Sort by chunk_index
            sorted_results = sorted(
                results,
                key=lambda x: x[0].metadata.get('chunk_index', 0)
            )
            return sorted_results
        except Exception as e:
            print(f"Error retrieving email chunks: {str(e)}")
            return []

    @staticmethod
    def _format_chunks(chunks: List[Document]) -> str:
        return "\n".join([
            f"Chunk {chunk.chunk_index + 1}/{chunk.metadata['total_chunks']}: {chunk.content}"
            for chunk in chunks
        ])