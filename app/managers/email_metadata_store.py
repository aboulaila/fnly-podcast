from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import String, DateTime, Integer, JSON, select, create_engine
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker

from app.models.email_metadata import EmailMetadata

Base = declarative_base()


class EmailMetadataModel(Base):
    __tablename__ = "email_metadata"

    email_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject: Mapped[str] = mapped_column(String)
    sender: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    headlines: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    links: Mapped[List[str]] = mapped_column(JSON)
    num_chunks: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    additional_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


class EmailMetadataStore:
    def __init__(self):
        self.session = self.initialize_db("postgresql://localhost:5432/fnly-podcast")

    @staticmethod
    def initialize_db(database_url: str):
        engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10
        )

        # Create tables
        Base.metadata.create_all(bind=engine)

        session = sessionmaker(
            engine,
            expire_on_commit=False
        )

        return session()

    def store_metadata(self, metadata: EmailMetadata):
        """Store email metadata in database"""
        try:
            metadata_dict = metadata.to_dict()
            db_metadata = EmailMetadataModel(
                email_id=metadata.email_id,
                subject=metadata.subject,
                sender=metadata.sender,
                timestamp=metadata.timestamp,
                headlines=metadata.headlines,
                content=metadata.content,
                links=metadata.links,
                num_chunks=metadata.num_chunks,
                total_tokens=metadata.total_tokens,
                additional_metadata=metadata_dict
            )

            self.session.add(db_metadata)
            self.session.commit()

        except Exception as e:
            self.session.rollback()
            print(f"Error storing metadata: {str(e)}")
            raise

    def get_metadata(self, email_id: str) -> Optional[EmailMetadata]:
        """Retrieve email metadata from database"""
        try:
            result = self.session.get(EmailMetadataModel, email_id)

            if result and result.additional_metadata:
                return EmailMetadata(**result.additional_metadata)
            return None

        except Exception as e:
            print(f"Error retrieving metadata: {str(e)}")
            return None

    def update_metadata(self, metadata: EmailMetadata):
        """Update existing metadata in database"""
        try:
            result = self.session.get(EmailMetadataModel, metadata.email_id)
            if result:
                metadata_dict = metadata.to_dict()
                result.subject = metadata.subject
                result.sender = metadata.sender
                result.timestamp = metadata.timestamp
                result.headlines = metadata.headlines
                result.content = metadata.content
                result.links = metadata.links
                result.num_chunks = metadata.num_chunks
                result.total_tokens = metadata.total_tokens
                result.additional_metadata = metadata_dict

                self.session.commit()
            else:
                self.store_metadata(metadata)

        except Exception as e:
            self.session.rollback()
            print(f"Error updating metadata: {str(e)}")
            raise

    def list_emails(self,
                    filter_dict: dict = None,
                    limit: int = 100,
                    offset: int = 0) -> List[EmailMetadata]:
        """List all email metadata matching the filter"""
        try:
            stmt = select(EmailMetadataModel)

            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    if hasattr(EmailMetadataModel, key):
                        column = getattr(EmailMetadataModel, key)
                        if isinstance(value, str):
                            conditions.append(column.ilike(f"%{value}%"))
                        elif value is None:
                            conditions.append(column.is_(None))
                        else:
                            conditions.append(column == value)

                if conditions:
                    stmt = stmt.where(*conditions)

            stmt = stmt.limit(limit).offset(offset)
            result = self.session.execute(stmt)

            return [
                EmailMetadata(**row.additional_metadata)
                for row in result.scalars()
                if row.additional_metadata
            ]

        except Exception as e:
            print(f"Error listing emails: {str(e)}")
            return []