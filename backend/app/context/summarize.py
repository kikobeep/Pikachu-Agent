from abc import ABC, abstractmethod


class ContextSummarizer(ABC):
    @abstractmethod
    async def summarize(
        self,
        previous_summary: RollingConversationSummary | None = None,
        history_message: tuple[Message,...]
    ) -> SummaryGenerationResult:
    

