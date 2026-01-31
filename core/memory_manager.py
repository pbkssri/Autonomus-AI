"""
Memory Management System for Autonomous AI Agent
Handles persistent storage, knowledge base, and conversation history
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path
import json
import logging
import hashlib
import sqlite3
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Memory entry representation"""
    id: str
    content: str
    category: str
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "metadata": self.metadata,
            "tags": self.tags
        }


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage"""
    
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> bool:
        """Add a memory entry"""
        pass
    
    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries"""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        pass
    
    @abstractmethod
    async def clear(self, category: Optional[str] = None) -> int:
        """Clear memory entries"""
        pass


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-based memory storage"""
    
    def __init__(self, db_path: str = "./data/memory.db"):
        """Initialize SQLite memory store
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    metadata TEXT,
                    tags TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON memories(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created 
                ON memories(created_at)
            """)
    
    async def add(self, entry: MemoryEntry) -> bool:
        """Add a memory entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, category, created_at, accessed_at, 
                     access_count, importance, metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.content,
                    entry.category,
                    entry.created_at.isoformat(),
                    entry.accessed_at.isoformat(),
                    entry.access_count,
                    entry.importance,
                    json.dumps(entry.metadata),
                    json.dumps(entry.tags)
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False
    
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (entry_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_entry(row)
                return None
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return None
    
    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries by content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT * FROM memories 
                       WHERE content LIKE ? OR category LIKE ? OR tags LIKE ?
                       ORDER BY importance DESC, access_count DESC
                       LIMIT ?""",
                    (f"%{query}%", f"%{query}%", f"%{query}%", limit)
                )
                
                return [self._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE id = ?", (entry_id,)
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    
    async def clear(self, category: Optional[str] = None) -> int:
        """Clear memory entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if category:
                    cursor = conn.execute(
                        "DELETE FROM memories WHERE category = ?", (category,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM memories")
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return 0
    
    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            category=row[2],
            created_at=datetime.fromisoformat(row[3]),
            accessed_at=datetime.fromisoformat(row[4]),
            access_count=row[5],
            importance=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            tags=json.loads(row[8]) if row[8] else []
        )


class MemoryManager:
    """Main memory management system"""
    
    def __init__(self, storage_path: str = "./data"):
        """Initialize memory manager
        
        Args:
            storage_path: Path to store memory data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.short_term: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.long_term = SQLiteMemoryStore(str(self.storage_path / "memory.db"))
        
        # Memory categories
        self.CATEGORY_CONVERSATION = "conversation"
        self.CATEGORY_KNOWLEDGE = "knowledge"
        self.CATEGORY_TASK = "task"
        self.CATEGORY_PREFERENCE = "preference"
        self.CATEGORY_FACT = "fact"
        
        # LRU cache size for short-term memory
        self.max_short_term = 100
    
    async def remember(self, content: str, category: str, 
                      importance: float = 0.5, 
                      metadata: Optional[Dict] = None,
                      tags: Optional[List[str]] = None) -> str:
        """Store a memory
        
        Args:
            content: Content to remember
            category: Memory category
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Memory entry ID
        """
        import hashlib
        import uuid
        
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()
        entry_id = str(uuid.uuid4())
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            category=category,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            importance=importance,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Add to short-term memory
        self.short_term[entry_id] = entry
        
        # Move to long-term if important or cache full
        if importance > 0.7 or len(self.short_term) > self.max_short_term:
            await self.long_term.add(entry)
        
        # Evict old entries from short-term
        while len(self.short_term) > self.max_short_term:
            oldest = next(iter(self.short_term))
            del self.short_term[oldest]
        
        return entry_id
    
    async def recall(self, query: str, category: Optional[str] = None) -> List[MemoryEntry]:
        """Recall memories matching query
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            List of matching memories
        """
        # Search long-term storage
        memories = await self.long_term.search(query, limit=20)
        
        # Update access information
        for memory in memories:
            memory.access_count += 1
            memory.accessed_at = datetime.now()
            await self.long_term.add(memory)
        
        return memories
    
    async def get_recent(self, limit: int = 10, 
                        category: Optional[str] = None) -> List[MemoryEntry]:
        """Get recent memories
        
        Args:
            limit: Maximum number of memories
            category: Optional category filter
            
        Returns:
            List of recent memories
        """
        try:
            with sqlite3.connect(self.long_term.db_path) as conn:
                query = "SELECT * FROM memories"
                params = []
                
                if category:
                    query += " WHERE category = ?"
                    params.append(category)
                
                query += " ORDER BY accessed_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                return [self.long_term._row_to_entry(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []
    
    async def forget(self, entry_id: str) -> bool:
        """Forget a memory
        
        Args:
            entry_id: Memory entry ID
            
        Returns:
            True if successful
        """
        # Remove from short-term if present
        if entry_id in self.short_term:
            del self.short_term[entry_id]
        
        # Remove from long-term
        return await self.long_term.delete(entry_id)
    
    async def clear_category(self, category: str) -> int:
        """Clear all memories in a category
        
        Args:
            category: Category to clear
            
        Returns:
            Number of entries cleared
        """
        count = await self.long_term.clear(category)
        
        # Also clear from short-term
        to_remove = [k for k, v in self.short_term.items() if v.category == category]
        for k in to_remove:
            del self.short_term[k]
        
        return count
    
    async def save_conversation(self, conversation_id: str, 
                               messages: List[Dict[str, str]]) -> None:
        """Save a conversation
        
        Args:
            conversation_id: Conversation identifier
            messages: List of messages
        """
        content = json.dumps({
            "conversation_id": conversation_id,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        })
        
        await self.remember(
            content=content,
            category=self.CATEGORY_CONVERSATION,
            importance=0.6,
            metadata={"conversation_id": conversation_id},
            tags=["conversation", conversation_id]
        )
    
    async def load_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        """Load a conversation
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of messages or None
        """
        memories = await self.recall(conversation_id, self.CATEGORY_CONVERSATION)
        
        for memory in memories:
            if memory.metadata.get("conversation_id") == conversation_id:
                try:
                    data = json.loads(memory.content)
                    return data.get("messages", [])
                except json.JSONDecodeError:
                    continue
        
        return None
    
    async def store_fact(self, fact: str, source: Optional[str] = None) -> str:
        """Store a factual statement
        
        Args:
            fact: Factual statement
            source: Source of the fact
            
        Returns:
            Memory entry ID
        """
        return await self.remember(
            content=fact,
            category=self.CATEGORY_FACT,
            importance=0.8,
            metadata={"source": source},
            tags=["fact"]
        )
    
    async def search_facts(self, query: str) -> List[str]:
        """Search for factual information
        
        Args:
            query: Search query
            
        Returns:
            List of matching facts
        """
        memories = await self.recall(query, self.CATEGORY_FACT)
        return [m.content for m in memories]
    
    def get_stats(self) -> Dict:
        """Get memory statistics
        
        Returns:
            Dictionary with memory stats
        """
        return {
            "short_term_count": len(self.short_term),
            "max_short_term": self.max_short_term,
            "categories": {
                "conversation": sum(1 for m in self.short_term.values() if m.category == self.CATEGORY_CONVERSATION),
                "knowledge": sum(1 for m in self.short_term.values() if m.category == self.CATEGORY_KNOWLEDGE),
                "task": sum(1 for m in self.short_term.values() if m.category == self.CATEGORY_TASK),
                "fact": sum(1 for m in self.short_term.values() if m.category == self.CATEGORY_FACT)
            }
        }
