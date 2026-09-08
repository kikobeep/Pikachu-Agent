class Conversation(BaseModel):
    """一个可恢复的本地聊天会话。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(default=0, ge=0)