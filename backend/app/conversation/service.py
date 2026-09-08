import asyncio



class ConversationService:
    def __init__(
        self,
        conversation_store: SQLiteConversationStore, # 加载会话历史
        conversation_summary_store: SQLiteConversationSummaryStore,
        run_manager: RunManager
    ):
        self.conversation_store = conversation_store
        self.conversation_summary_store = conversation_summary_store
        self.run_manager = run_manager
        self._locks : dict[str,Any] = {}

    # 每一个conversation都需要上锁
    def _lock(self,conversation_id):
        lock = self._locks.get(conversation_id)
        if lock is None: # 如果当前会话还没有加锁，需要加上线程锁
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock
    
    def _run(
        self,
        conversation_id: str
        
        
    ):
    # 1) 从持久化源加载“触发那一刻最新”的 history / summary。
    history: tuple[Any, ...] = ()
    if conversation_id is not None:
        history = tuple(
            await self._conversation_store.load_messages(conversation_id)
        )



