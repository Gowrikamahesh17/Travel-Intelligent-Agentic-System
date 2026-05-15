"""
Database utilities and session management.
Provides singleton pattern for database access and utilities.
"""

from typing import Optional, Generator
from sqlalchemy.orm import Session
from contextlib import contextmanager
from src.common import get_logger
from .models import init_db, get_db_engine

logger = get_logger(__name__)


class DatabaseManager:
    """Singleton database manager for session management."""

    _instance: Optional["DatabaseManager"] = None
    _session_factory = None
    _engine = None

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, database_url: str) -> None:
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL
        """
        if self._session_factory is None:
            logger.info(f"Initializing database: {database_url}")
            self._session_factory = init_db(database_url)
            self._engine = get_db_engine(database_url)
            logger.info("Database initialized successfully")

    def get_session(self) -> Session:
        """
        Get new database session.

        Returns:
            SQLAlchemy Session

        Raises:
            RuntimeError: If database not initialized
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory()

    @contextmanager
    def session_context(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Usage:
            with db_manager.session_context() as session:
                user = session.query(UserProfile).first()

        Yields:
            Database session
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session() -> Session:
    """Get new database session (convenience function)."""
    return get_db_manager().get_session()
