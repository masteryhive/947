from src.config.settings import settings
from sqlalchemy import Column, String, Float, DateTime, Text, UUID, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()

class InsurancePolicy(Base):
    __tablename__ = "insurance_policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_number = Column(String(100), nullable=False, unique=True)
    insured_name = Column(String(255), nullable=False)
    sum_insured = Column(Float, nullable=False)
    premium = Column(Float, nullable=False)
    own_retention_ppn = Column(Float, nullable=True)
    own_retention_sum_insured = Column(Float, nullable=True)
    own_retention_premium = Column(Float, nullable=True)
    treaty_ppn = Column(Float, nullable=True)
    treaty_sum_insured = Column(Float, nullable=True)
    treaty_premium = Column(Float, nullable=True)
    insurance_period_start_date = Column(DateTime, nullable=False)
    insurance_period_end_date = Column(DateTime, nullable=False)

    # Vector storage for semantic search
    content_vector = Column(Vector(settings.VECTOR_DIMENSION))  # Based on vector dimension
    searchable_content = Column(Text)
    policy_metadata = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index('idx_policy_number', policy_number),
        Index('idx_insured_name', insured_name),
        Index('idx_insurance_period', insurance_period_start_date, insurance_period_end_date),
        Index('idx_content_vector', content_vector, postgresql_using='ivfflat', postgresql_ops={'content_vector': 'vector_cosine_ops'}),
        Index('idx_policy_metadata', policy_metadata, postgresql_using='gin'),
    )

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.VECTOR_DIMENSION))
    document_metadata = Column(JSONB)
    document_type = Column(String(50))
    user_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_document_embedding', embedding, postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('idx_document_metadata', document_metadata, postgresql_using='gin'),
        Index('idx_user_id', user_id),
    )
