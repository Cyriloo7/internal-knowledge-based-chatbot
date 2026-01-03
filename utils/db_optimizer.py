"""
Database optimization utilities
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, event
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

def create_indexes(db, models):
    """Create database indexes for optimized queries"""
    indexes_created = []
    
    try:
        # Indexes for User model
        if hasattr(models, 'User'):
            indexes_created.append(
                Index('idx_user_username', models.User.username)
            )
            indexes_created.append(
                Index('idx_user_email', models.User.email)
            )
            indexes_created.append(
                Index('idx_user_role', models.User.role)
            )
        
        # Indexes for Conversation model
        if hasattr(models, 'Conversation'):
            indexes_created.append(
                Index('idx_conversation_user_id', models.Conversation.user_id)
            )
            indexes_created.append(
                Index('idx_conversation_updated_at', models.Conversation.updated_at)
            )
            indexes_created.append(
                Index('idx_conversation_user_updated', 
                      models.Conversation.user_id, 
                      models.Conversation.updated_at)
            )
        
        # Indexes for ChatHistory model
        if hasattr(models, 'ChatHistory'):
            indexes_created.append(
                Index('idx_chathistory_user_id', models.ChatHistory.user_id)
            )
            indexes_created.append(
                Index('idx_chathistory_timestamp', models.ChatHistory.timestamp)
            )
            indexes_created.append(
                Index('idx_chathistory_collection', models.ChatHistory.collection_name)
            )
            indexes_created.append(
                Index('idx_chathistory_user_timestamp', 
                      models.ChatHistory.user_id, 
                      models.ChatHistory.timestamp)
            )
            indexes_created.append(
                Index('idx_chathistory_conversation', models.ChatHistory.conversation_id)
            )
        
        # Indexes for Document model
        if hasattr(models, 'Document'):
            indexes_created.append(
                Index('idx_document_collection', models.Document.collection_name)
            )
            indexes_created.append(
                Index('idx_document_filename', models.Document.filename)
            )
            indexes_created.append(
                Index('idx_document_category', models.Document.category_id)
            )
        
        # Indexes for UserSettings model
        if hasattr(models, 'UserSettings'):
            indexes_created.append(
                Index('idx_usersettings_user_id', models.UserSettings.user_id, unique=True)
            )
        
        logger.info(f"Created {len(indexes_created)} database indexes")
        return indexes_created
        
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        return []

def optimize_connection_pool(db):
    """Optimize database connection pool settings"""
    try:
        # Configure connection pool for better performance
        engine = db.engine
        
        # Set pool size and max overflow
        engine.pool.size = 10  # Number of connections to maintain
        engine.pool.max_overflow = 20  # Maximum overflow connections
        engine.pool.pool_timeout = 30  # Timeout for getting connection
        engine.pool.recycle = 3600  # Recycle connections after 1 hour
        
        logger.info("Database connection pool optimized")
        return True
    except Exception as e:
        logger.error(f"Error optimizing connection pool: {str(e)}")
        return False

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas for better performance"""
    if 'sqlite' in str(dbapi_conn):
        cursor = dbapi_conn.cursor()
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Increase cache size
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")
        # Optimize for speed
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

def analyze_query_performance(db, query):
    """Analyze query performance (for debugging)"""
    try:
        if 'sqlite' in str(db.engine.url):
            # SQLite EXPLAIN QUERY PLAN
            result = db.session.execute(f"EXPLAIN QUERY PLAN {query}")
            return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        return []

