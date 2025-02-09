import logging
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ai_services.email_summarizer import EmailSummarizer
from app.models.email_content import EmailContent
from app.models.email_metadata import EmailMetadata
from app.services.email_metadata_store import EmailMetadataStore


class EmailVectorStore:
    def __init__(self, vector_store: PineconeVectorStore, metadata_store: EmailMetadataStore):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.max_tokens = 100

        self.summarizer = EmailSummarizer()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.logger = logging.getLogger(__name__)

    def process_and_store_email(self, email_content: EmailContent) -> EmailMetadata:
        """Process and store email content with proper metadata handling"""
        email_id = str(uuid.uuid4())

        metadata = EmailMetadata(
            email_id=email_id,
            subject=email_content.subject,
            sender=email_content.sender,
            timestamp=email_content.received_date,
            headlines="",
            content=email_content.processed_text,
            links=email_content.links,
            num_chunks=0,
            total_tokens=0
        )

        try:
            self.metadata_store.store_metadata(metadata)

            chunks = self._create_chunks(email_content.processed_text)
            self._store_chunks(chunks, metadata)

            metadata.num_chunks = len(chunks)
            metadata.total_tokens = sum(len(chunk.split()) for chunk in chunks)
            self.metadata_store.update_metadata(metadata)

        except Exception as e:
            self.logger.error(f"Error in process_and_store_email: {e}")
            raise

        return metadata

    def get_email_chunks(self, email_id: str, k: int = 5):
        """Retrieve all chunks for a specific email"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query="",
                filter={"email_id": email_id},
                k=k
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

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks using LangChain's text splitter."""
        try:
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _store_chunks(self, chunks: List[str], email_metadata: EmailMetadata) -> None:
        """Store chunks with properly formatted metadata"""
        try:
            documents = []

            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **{key: value for key, value in email_metadata.to_dict().items() if key != "links"}
                }

                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)

            self.vector_store.add_documents(documents)

        except Exception as e:
            self.logger.error(f"Error storing chunks in Vector Store: {str(e)}")
            raise