import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, JSON, create_engine
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker

from app.models.analysis_result import AnalysisResult

Base = declarative_base()


class EmailAnalysisModel(Base):
    __tablename__ = "email_analysis"

    analysis_id: Mapped[str] = mapped_column(String, primary_key=True)
    email_id: Mapped[str] = mapped_column(String)
    analysis_result: Mapped[dict] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


class EmailAnalysisStore:
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

    def store_analysis(self, analysis_result: AnalysisResult):
        try:
            analysis_dict = analysis_result.to_dict()
            db_analysis = EmailAnalysisModel(
                analysis_id=str(uuid.uuid4()),
                email_id=analysis_result.email_id,
                analysis_result=analysis_dict
            )

            self.session.add(db_analysis)
            self.session.commit()

            return db_analysis
        except Exception as e:
            self.session.rollback()
            print(f"Error storing analysis: {str(e)}")
            raise

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        try:
            result = self.session.get(EmailAnalysisModel, analysis_id)

            if result and result.analysis_result:
                return AnalysisResult(**result.analysis_result)
            return None

        except Exception as e:
            print(f"Error retrieving analysis: {str(e)}")
            return None

    def get_analysis_by_ids(self, analysis_ids: List[str]) -> List[Optional[AnalysisResult]]:
        """Retrieve multiple analysis results by a list of analysis IDs."""
        try:
            results = self.session.query(EmailAnalysisModel).filter(
                EmailAnalysisModel.analysis_id.in_(analysis_ids)
            ).all()

            return [AnalysisResult(**result.analysis_result) if result.analysis_result else None for result in results]

        except Exception as e:
            print(f"Error retrieving analyses: {str(e)}")
            return []

    def update_analysis(self, analysis_result: AnalysisResult):
        try:
            result = self.session.get(EmailAnalysisModel, analysis_result.analysis_id)
            if result:
                analysis_dict = analysis_result.to_dict()
                result.analysis_result = analysis_dict

                self.session.commit()
            else:
                self.store_analysis(analysis_result)

        except Exception as e:
            self.session.rollback()
            print(f"Error updating analysis: {str(e)}")
            raise