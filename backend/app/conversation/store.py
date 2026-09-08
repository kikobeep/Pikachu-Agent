import secrets
import aiosqlite

DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[2] / ".vesta" / "vesta.db"
)

_CONVERSATION_SELECT = """
SELECT
    c.id,
    c.title,
    c.created_at,
    c.updated_at,
    COUNT(m.id) AS message_count
FROM conversations AS c
LEFT JOIN messages AS m ON m.conversation_id = c.id
"""

class SQLiteConversationStore:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
    

    async def create(self,
        *,
        title: str = "新会话",
        messages: Sequence[Message] = (),
    ) -> Conversation:
     
        title = _normalize_title(title)
        conversation_id = secrets.token_hex(16)
        now = _now_iso()
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, normalized_title, now, now),
            )
            await self._insert_messages(db,conversation_id,messages,now)
            await database.commit()
        
        conversation = await self.get(conversation_id)
        
        if conversation is None:  # pragma: no cover - SQLite 写入后的防御性检查
            raise RuntimeError("创建会话后无法重新读取会话")
        return self._form_conversation(conversation)
    
    def _insert_messages(self,
            db: aiosqlite.Connection,
            conversation_id: str,
            messages: Sequence[Message] = (),
            created_time: str
        ):
        for index, message in enumerate(messages):
            rows.append(
                (
                    conversation_id,
                    index,
                    message.role,
                    message.content,
                    message.name,
                    message.too_call_id,
                    json.dumps(
                        [
                            tool_call.model_dump(mode="json")
                            for tool_call in message.tool_calls
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    created_time
                )
            )
        if rows:
            await database.executemany(
                """
                INSERT INTO messages (
                    conversation_id,
                    sequence,
                    role,
                    content,
                    name,
                    tool_call_id,
                    tool_calls_json,
                    reasoning,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        async def get(self,conversation_id:str):
            async with aiosqlite.connect(self.database_path) as db:
                row = await database.execute(
                    _CONVERSATION_SELECT  + '''
                    where c.id = ? GROUP BY c.id
                    ''',
                    (conversation_id,),
                ).fetchone()
            return row

        async def latest(self) -> Conversation | None:
            """返回最近更新的会话。"""

            conversations = await self.list(limit=1)
            return conversations[0] if conversations else None

        async def list(self, *, limit: int = 20) -> tuple[Conversation, ...]:
            """按最近更新时间倒序列出会话。"""

            if limit < 1:
                raise ValueError("limit must be at least 1")
            async with aiosqlite.connect(self.database_path) as db:
                cursor = await db.execute(
                    _CONVERSATION_SELECT
                    + " GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
            return tuple(_form_conversation(row) for row in rows) 
        
        async def load_messages(self, conversation_id: str) -> tuple[Message, ...]:
            """按原始顺序读取会话中的全部消息。"""

            if await self.get(conversation_id) is None:
                raise KeyError(f"会话不存在：{conversation_id}")
            async with aiosqlite.connect(self.database_path) as db:
                cursor = await db.execute(
                    """
                    SELECT role, content, name, tool_call_id, tool_calls_json, reasoning
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY sequence ASC
                    """,
                    (conversation_id,),
                ).fetchall()
            return tuple(_form_message(row) for row in rows)
        
        async def search_messages(
            self,
            conversation_id: str,
            query: str,
            *,
            limit: int = 10,
        ) -> tuple[Message, ...]:
            """在当前会话的原始消息中检索，不读取压缩后的模型请求视图。"""

            normalized = query.strip()
            if not normalized:
                raise ValueError("query must be a non-empty string")
            if limit < 1 or limit > 20:
                raise ValueError("limit must be between 1 and 20")
            if await self.get(conversation_id) is None:
                raise KeyError(f"会话不存在：{conversation_id}")
            async with aiosqlite.connect(self.database_path) as db:
                row = await database.execute(
                    """
                    SELECT sequence, role, content, name, tool_call_id,
                        tool_calls_json, reasoning, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    AND instr(lower(coalesce(content, '')), lower(?)) > 0
                    ORDER BY sequence DESC LIMIT ?
                    """,
                    (conversation_id, normalized, limit),
                ).fetchall()
            return tuple(_form_message(row) for row in rows)

        async def delete(self, conversation_id: str) -> bool:
            """删除会话及其消息，并返回是否实际删除。"""
            async with aiosqlite.connect(self.database_path) as db:
                row = await db.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                await db.commit()
            return row.rowcount > 0


        def _form_conversation(row: aiosqlite.Row) -> Conversation:
            return Conversation(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                message_count=row["message_count"],
            )
        
        def _form_message(row: aiosqlite.Row) -> Message:
            raw_tool_calls = json.loads(row["tool_calls_json"])
            return Message(
                role=row["role"],
                content=row["content"],
                name=row["name"],
                tool_call_id=row["tool_call_id"],
                tool_calls=tuple(ToolCall(item) for item in raw_tool_calls),
                reasoning=row["reasoning"],
            )

