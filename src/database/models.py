"""
Database models using SQLAlchemy ORM.
Defines UserProfile, QueryHistory, TripHistory, and VisaCache schemas.
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Boolean,
    JSON,
    Text,
    Float,
    ForeignKey,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional

Base = declarative_base()


class UserProfile(Base):
    """User profile with preferences, constraints, and consent."""

    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    nationality = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    
    # Preferences (JSON)
    travel_preferences = Column(
        JSON,
        default={
            "budget_range": {"min": 0, "max": 10000},
            "trip_duration": 7,
            "travel_pace": "moderate",
            "accommodation_type": "hotel",
            "dietary_restrictions": [],
        },
    )
    
    # Constraints
    mobility_constraints = Column(JSON, default={})  # e.g., wheelchair access
    health_conditions = Column(JSON, default={})  # Medical information
    visa_status = Column(JSON, default={})  # Current visa information
    
    # Consent & Settings
    consent_data_processing = Column(Boolean, default=False)
    consent_recommendations = Column(Boolean, default=False)
    consent_rag_learning = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    queries = relationship("QueryHistory", back_populates="user")
    trips = relationship("TripHistory", back_populates="user")

    def __repr__(self) -> str:
        return f"<UserProfile id={self.id} name={self.name}>"


class QueryHistory(Base):
    """Full query logs for analysis and pattern learning."""

    __tablename__ = "query_history"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), nullable=False)
    
    # Query content
    query_text = Column(Text, nullable=False)
    query_type = Column(String, nullable=False)  # e.g., "itinerary", "info"
    
    # Agent execution trace
    agents_involved = Column(JSON, default=[])  # List of agent names
    execution_trace = Column(JSON, default={})  # Full agent execution history
    reasoning_steps = Column(JSON, default=[])  # Step-by-step reasoning
    
    # Results
    result = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    xai_explanation = Column(JSON, nullable=True)  # XAI validator output
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    execution_time_ms = Column(Integer, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Feedback
    user_rating = Column(Integer, nullable=True)  # 1-5
    user_feedback = Column(Text, nullable=True)
    
    # Relationship
    user = relationship("UserProfile", back_populates="queries")

    def __repr__(self) -> str:
        return f"<QueryHistory id={self.id} user_id={self.user_id} type={self.query_type}>"


class TripHistory(Base):
    """Completed trips for pattern learning and personalization."""

    __tablename__ = "trip_history"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), nullable=False)
    
    # Trip details
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Trip composition
    itinerary = Column(JSON, default=[])  # List of activities/stops
    accommodations = Column(JSON, default=[])
    transportation = Column(JSON, default=[])
    
    # Experience & Feedback
    total_budget_spent = Column(Float, nullable=True)
    overall_rating = Column(Float, nullable=True)  # 1-5
    highlights = Column(JSON, default=[])
    challenges = Column(JSON, default=[])
    recommendations = Column(Text, nullable=True)
    
    # AI-Generated Insights
    ai_insights = Column(JSON, nullable=True)
    pattern_tags = Column(JSON, default=[])  # Tags for RAG learning
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    user = relationship("UserProfile", back_populates="trips")

    def __repr__(self) -> str:
        return f"<TripHistory id={self.id} destination={self.destination}>"


class VisaCache(Base):
    """Cache for visa requirements to avoid repeated API calls."""

    __tablename__ = "visa_cache"

    id = Column(String, primary_key=True, index=True)
    origin_country = Column(String, nullable=False, index=True)
    destination_country = Column(String, nullable=False, index=True)
    
    # Visa information
    visa_required = Column(Boolean, nullable=False)
    visa_type = Column(String, nullable=True)
    visa_description = Column(Text, nullable=True)
    visa_cost = Column(Float, nullable=True)
    
    # Requirements & Documents
    requirements = Column(JSON, default=[])
    documents_needed = Column(JSON, default=[])
    processing_time_days = Column(Integer, nullable=True)
    
    # Additional Info
    restrictions = Column(JSON, default=[])
    special_conditions = Column(JSON, default=[])
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_verified = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<VisaCache {self.origin_country} -> {self.destination_country}>"


# Database initialization utilities
def init_db(database_url: str) -> sessionmaker:
    """
    Initialize database and create tables.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        Session factory
    """
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def get_db_engine(database_url: str):
    """Get SQLAlchemy engine."""
    return create_engine(database_url, connect_args={"check_same_thread": False})
