
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.agent.budget import RunBudgetConfig
from app.agent.events import AgentEventHandler
from app.agent.runtime import AgentRuntime
from app.ace import AceCoordinator
from app.agent.spec import load_agent_prompt, resolve_agent_md_path
from app.checkpoint.store import CheckpointStore
from app.context import ContextSettings, ContextSummaryModelConfig
from app.context.manager import ContextManager
from app.context.reducers.conversation import ConversationReducer
from app.context.summarize import ModelContextSummarizer
from app.context.summary_store import ConversationSummaryStore
from app.context.handoff_store import HandoffStore
from app.conversation.service import ConversationService
from app.conversation.store import DEFAULT_DATABASE_PATH, ConversationStore
from app.conversation.tools import register_history_tools
from app.evidence.recorder import EvidenceRecorder
from app.evidence.store import EvidenceStore
from app.evidence.tools import register_evidence_tools
from app.memory import MemoryArchiveConfig, MemoryReflectionConfig
from app.memory.archive_model import ArchiveMemoryReflector
from app.memory.embedding import (
    EmbeddingAdapter,
    MemoryEmbeddingSettings,
    build_embedding_adapter,
)
from app.memory.manager import MemoryManager
from app.memory.search_config import MemorySearchConfig
from app.memory.reflection_model import MemoryReflector
from app.memory.store import DEFAULT_MEMORY_DIR
from app.memory.tools import register_memory_tools, register_memory_write_tools
from app.model import ModelConfig, ModelProvider, ModelAdapterRegistry
from app.run.manager import RunManager
from app.run.store import RunStore
from app.sandbox.supervisor import SandboxSupervisor
from app.skills import DEFAULT_PROJECT_SKILLS_DIR, DEFAULT_USER_SKILLS_DIR
from app.skills.context import SkillContextProvider
from app.skills.store import SkillStore
from app.skills.tools import register_skill_tools
from app.skills_improve.candidate_store import SkillCandidateStore
from app.skills_improve.config import SkillImprovingSettings
from app.skills_improve.distiller import ModelMultiTeacherDistiller, ModelProcedureDistiller
from app.skills_improve.evidence import DefaultEventSelector, TraceEvidenceBuilder
from app.skills_improve.miner import ModelPatternMiner
from app.skills_improve.service import SkillImproveService
from app.plan.attribution import PlanToolOutputAttributionResolver
from app.plan.context import PlanContextProvider
from app.plan.store import PlanStore
from app.plan.tools import register_plan_tools
from app.plan.runner import PlanRunner
from app.tools.builtin import builtin_tool_registry
from app.tools.builtin._workspace import workspace_root_path
from app.tools.approval import ApprovalGate
from app.tools.permissions.policy import PermissionPolicyEngine
from app.tools.permissions.store import PermissionRuleStore
from app.tools.register import ToolRegistry
from app.trace.store import TraceStore
from app.plan import DEFAULT_PLANS_DIR
from app.skills.config import SkillSettings
from app.memory.tools import DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES


_ON_DEMAND_TOOL_NAMES = frozenset(
    {
        "http_request",
        *DEFAULT_ON_DEMAND_MEMORY_TOOL_NAMES,
    }
)

def _mark_demand_tools(
    registry: ToolRegistry,
    names: frozenset[str],
) -> None:
    """把不常用工具标记为按需暴露（不进入模型 schema，可 tool_search 激活）。"""

    for name in names:
        tool = registry.unregister(name)
        registry.register(tool, on_demand=True)

class Application:

    def __init__(
        self,
        *,
        provider: ModelProvider | str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        agent_md: str | Path | None = None,
        database: str | Path = DEFAULT_DATABASE_PATH,
        plans_dir: str | Path = DEFAULT_PLANS_DIR,
        mcp_config: str | Path = None,
        memory_dir: str | Path | None = None,
        skills_user_dir: str | Path | None = None,
        skills_project_dir: str | Path | None = None,
        max_steps: int = 12,
        max_tool_rounds: int = 15,
        max_output_tokens: int | None = None,
        run_budget_config: RunBudgetConfig | None = None,
        settings: ModelConfig | None = None,
        registry: ModelAdapterRegistry | None = None,
        shared_event_handler: AgentEventHandler | None = None,
        memory_reflection_config: MemoryReflectionConfig | None = None,
        memory_archive_config: MemoryArchiveConfig | None = None,
        context_summary_config: ContextSummaryModelConfig | None = None,
        skill_improving_settings: SkillImprovingSettings | None = None,
        approval_gate: ApprovalGate | None = None,
        desktop_approval: bool = False,
        # computer_runtime: ComputerRuntime | None = None,
        # computer_host_status: ComputerHostStatus | None = None,
        workspace_root: str | Path | None = None,
        enable_memory_write_tools: bool = False,
        context_summary_enabled: bool = True,
        memory_search_vector_only: bool = False,
        memory_search_min_similarity: float = 0.0,
        memory_search_top_k: int = 5,
        memory_search_vector_top_k: int = 1,
        memory_search_text_top_k: int = 1,
        memory_auto_search_enabled: bool = True,
        ace_enabled: bool = False,
        ace_data_dir: str | Path | None = None,
        ace_playbook_path: str | Path | None = None,
        ace_max_strategies: int = 6,
        ace_select_top_k: int = 1,
        ace_irrelevant_streak_limit: int = 5,
        ace_max_output_tokens: int = 1800,
        ace_bulletpoint_analyzer_enabled: bool = True,
        ace_bulletpoint_analyzer_threshold: float = 0.90,
        ace_max_bullets: int = 16,
        ace_harmful_prune_threshold: int = 3,
    ) -> None:
        self.database = Path(database).expanduser().resolve()
        self.plans_dir = Path(plans_dir).expanduser().resolve()
        self.workspace_root = workspace_root_path(workspace_root)
        self.enable_memory_write_tools = enable_memory_write_tools
        self.context_summary_enabled = context_summary_enabled
        self.memory_search_vector_only = memory_search_vector_only
        self.memory_search_min_similarity = memory_search_min_similarity
        self.memory_search_top_k = memory_search_top_k
        self.memory_search_vector_top_k = memory_search_vector_top_k
        self.memory_search_text_top_k = memory_search_text_top_k
        self.memory_auto_search_enabled = memory_auto_search_enabled
        self.ace_enabled = ace_enabled
        self.ace_data_dir = (
            Path(ace_data_dir).expanduser().resolve()
            if ace_data_dir is not None else self.database.parent / "ace"
        )
        self.ace_playbook_path = (
            Path(ace_playbook_path).expanduser().resolve()
            if ace_playbook_path is not None
            else self.ace_data_dir / "playbook.txt"
        )
        self.ace_max_strategies = ace_max_strategies
        self.ace_select_top_k = ace_select_top_k
        self.ace_irrelevant_streak_limit = ace_irrelevant_streak_limit
        self.ace_max_output_tokens = ace_max_output_tokens
        self.ace_bulletpoint_analyzer_enabled = ace_bulletpoint_analyzer_enabled
        self.ace_bulletpoint_analyzer_threshold = ace_bulletpoint_analyzer_threshold
        self.ace_max_bullets = ace_max_bullets
        self.ace_harmful_prune_threshold = ace_harmful_prune_threshold
        # self.mcp_config = Path(mcp_config).expanduser().resolve()
        # self.mcp_config_store = MCPConfigurationStore(self.mcp_config)
        self.memory_dir = (
            Path(memory_dir).expanduser().resolve() if memory_dir is not None else None
        )
        self.skills_user_dir = (
            Path(skills_user_dir).expanduser().resolve()
            if skills_user_dir is not None
            else None
        )
        self.skills_project_dir = (
            Path(skills_project_dir).expanduser().resolve()
            if skills_project_dir is not None
            else None
        )
      
        if system_prompt is not None:
            self.system_prompt = system_prompt
            self.agent_md_path: Path | None = None
        else:
            self.system_prompt = load_agent_prompt(
                explicit_path=agent_md,
                workspace_root=self.workspace_root,
            )
            self.agent_md_path = resolve_agent_md_path(
                explicit_path=agent_md,
                workspace_root=self.workspace_root,
            )
        self.max_steps = max_steps
        self.max_tool_rounds = max_tool_rounds
        self.max_output_tokens = max_output_tokens
        self._run_budget_config = run_budget_config
        self._memory_reflection_config = memory_reflection_config
        self._memory_archive_config = memory_archive_config
        self._context_summary_config = context_summary_config
        self._skill_improving_settings = skill_improving_settings
        # True = DesktopApprovalGate（Host）；False = ConsoleApprovalGate（CLI）。
        # self.desktop_approval = desktop_approval
        # Computer Runtime 由入口注入；None 时不注册 computer_* 工具。
        # self._computer_runtime = computer_runtime
        # Computer Host 状态（bootstrap 产物；None = 未配置 Computer）。
        # self.computer_host_status = computer_host_status
        # 未传入配置时，从 backend/.config 和环境变量加载。
        self.settings = settings or ModelConfig()
        self.active_model_roles: dict[str, dict[str, object]] = {}
        self.host_restart_callback: Callable[[], None] | None = None
        if registry is not None:
            self.registry = registry
            self.provider: str = (
                provider.value if isinstance(provider, ModelProvider) else provider
            ) or "fake"
            self.model: str = model or "fake-model"
        else:
            self.registry = ModelAdapterRegistry(self.settings)
            selected = (
                ModelProvider(provider)
                if provider is not None
                else self.settings.model_default_provider
            )
            self.provider = selected.value
            config = self.settings.load_provider_config(selected)
            self.model = model or config.model

        # 全局共享事件观察者（Server 在 start() 前注入 Desktop 广播 handler）。
        self.shared_event_handler = shared_event_handler
        # 审批门：宿主层（CLI/Desktop/eval）注入；None 时 ToolExecutor 兜底 DenyAllGate。
        self.approval_gate = approval_gate

        # Post-Run 后台任务（Memory Reflection 等）由 Application 统一管理生命周期。
        # self.post_run_processor = PostRunProcessor()

        # 在 start() 中构建的依赖。
        self.conversation_store: ConversationStore | None = None
        self.summary_store: ConversationSummaryStore | None = None
        self.evidence_store: EvidenceStore | None = None
        self.trace_store: TraceStore | None = None
        self.checkpoint_store: CheckpointStore | None = None
        self.rule_store: PermissionRuleStore | None = None
        self.policy_engine: PermissionPolicyEngine | None = None
        # self.approval_store: SQLiteApprovalStore | None = None
        # self.approval_gate: Any | None = None
        # self.desktop_approval_gate: DesktopApprovalGate | None = None
        # self.artifact_store: SQLiteArtifactStore | None = None
        # # self.artifact_service: ArtifactService | None = None
        # self.computer_runtime: ComputerRuntime | None = None
        # self.computer_lease: ComputerLeaseManager | None = None
        # self.computer_session: ComputerSessionManager | None = None
        self.tool_registry: ToolRegistry | None = None
        self.plan_store: PlanStore | None = None
        self.memory_manager: MemoryManager | None = None
        self.memory_embedding_adapter: EmbeddingAdapter | None = None
        self.skill_store: SkillStore | None = None
        self.skill_context_provider: SkillContextProvider | None = None
        self.skill_candidate_store: SkillCandidateStore | None = None
        self.skill_improving: SkillImproveService | None = None
        self.memory_reflector: MemoryReflector | None = None
        self.memory_archive_reflector: ArchiveMemoryReflector | None = None
        self.memory_reflection_enabled = True
        self.memory_archive_enabled = True
        self.context_summarizer: ModelContextSummarizer | None = None
        self.context_manager: ContextManager | None = None
        self.ace: AceCoordinator | None = None
        # self.mcp_manager: MCPClientManager | None = None
        self.mcp_statuses: tuple[Any, ...] = ()
        self.mcp_error: str | None = None
        self.runtime: AgentRuntime | None = None
        self.run_store: RunStore | None = None
        self.run_manager: RunManager | None = None
        self.conversation_service: ConversationService | None = None
        # self.automation_store: SQLiteAutomationStore | None = None
        # self.automation_scheduler: AutomationScheduler | None = None
        self.reconciled_runs: tuple[Any, ...] = ()

        self._started = False

    async def start(self):
        if self._started:
            return

        database = self.database
        sandbox_supervisor = SandboxSupervisor(self.workspace_root)
        tool_registry = builtin_tool_registry(
            self.workspace_root,
            sandbox_supervisor=sandbox_supervisor,
        )
        '''
        conversation_store: 会话信息，以及用户消息、模型回复、工具调用和结果。Run 开始前读取历史；结束后保存更新后的消息
        '''
        conversation_store = ConversationStore(database)
        await conversation_store.initialize()
        register_history_tools(tool_registry, conversation_store)

        '''
        plan_store: 计划目标、约束、步骤、进度、关键事实等。通过plan_create、plan_update、plan_get 等工具创建、修改或读取计划
        '''
        plan_store = PlanStore(self.plans_dir)
        await plan_store.initialize()
        register_plan_tools(tool_registry, plan_store)

        '''
        evidence_store: 保存工具原始输出的存储对象，方便后续重新查阅，包括原始输出正文、来自哪个工具、哪次工具调用
        '''
        evidence_store = EvidenceStore(database)
        await evidence_store.initialize()
        register_evidence_tools(tool_registry, evidence_store)
        evidence_recorder = EvidenceRecorder(
            evidence_store,
            attribution_resolver=PlanToolOutputAttributionResolver(plan_store),
        )
        '''
        summary_store：滚动摘要，以及摘要覆盖了多少条原始消息
        '''
        summary_store = ConversationSummaryStore(database)
        await summary_store.initialize()
        handoff_store = HandoffStore(database.parent / "handoffs")
        await handoff_store.initialize()
        
        '''
        保存agent_runs：Run 的整体信息：状态、模型、开始／结束时间、步数、Token 用量等 和agent_events：按顺序发生的事件：模型请求、工具执行、审批、预算告警、运行结束等
        '''
        trace_store = TraceStore(database)
        await trace_store.initialize()
        '''
        checkpoint_store: Run 的执行状态、当前阶段、步数、待执行工具、已完成工具结果。Run 开始、请求模型前、执行工具前后、结束或中断时
        '''
        checkpoint_store = CheckpointStore(database)
        await checkpoint_store.initialize()
        
        '''
        rule_store: 工具的允许／拒绝规则、参数匹配条件、适用范围。工具执行前匹配规则（ALLOW/deny/ask）；用户选择记住审批决定时保存规则
        '''
        rule_store = PermissionRuleStore(database)
        await rule_store.initialize()

        policy_engine = PermissionPolicyEngine(rule_store)

        '''
        Agent 明确交付给用户的成果，例如生成的报告、Excel 文件、图片，或者一个网页链接
        '''
        # artifact_store = SQLiteArtifactStore(database)
        # await artifact_store.initialize()
        # artifact_service = ArtifactService(
        #     artifact_store,
        #     self.workspace_root,
        #     managed_dir=database.parent / "artifacts",
        # )
        # register_artifact_tools(tool_registry, artifact_service)
        
       
        
        '''
        记忆
        '''
        memory_embedding_settings = MemoryEmbeddingSettings()
        memory_embedding_adapter = build_embedding_adapter(
            memory_embedding_settings
        )
        self.memory_embedding_adapter = memory_embedding_adapter
        memory_manager = MemoryManager(
            memory_dir=self.memory_dir or DEFAULT_MEMORY_DIR,
            embedding=memory_embedding_adapter,
            search_configs=MemorySearchConfig(
                top_k=self.memory_search_top_k,
                vector_top_k=self.memory_search_vector_top_k,
                text_top_k=self.memory_search_text_top_k,
                vector_only=self.memory_search_vector_only,
                min_vector_similarity=self.memory_search_min_similarity,
            ),
            min_vector_similarity=memory_embedding_settings.min_similarity,
        )
        await memory_manager.initialize()
        register_memory_tools(tool_registry, memory_manager)
        if self.enable_memory_write_tools:
            register_memory_write_tools(tool_registry, memory_manager)

        '''
        skill
        '''
        skill_store = SkillStore(
            user_dir=self.skills_user_dir or DEFAULT_USER_SKILLS_DIR,
            project_dir=self.skills_project_dir or DEFAULT_PROJECT_SKILLS_DIR,
        )
        await skill_store.initialize()
        register_skill_tools(tool_registry, skill_store)
        skill_settings = SkillSettings()
        skill_context_provider = SkillContextProvider(
            max_tokens=skill_settings.skill_context_max_tokens,
            max_active=skill_settings.skill_max_active,
            max_meta_tokens=skill_settings.skill_catalog_max_tokens,
        )

     
        _mark_demand_tools(tool_registry, _ON_DEMAND_TOOL_NAMES)

        reflection_config = self._memory_reflection_config or MemoryReflectionConfig()
        memory_reflector = MemoryReflector(
            self.registry.get(self.provider),
            model=self.model,
            config=reflection_config,
        )
        archive_config = (
            self._memory_archive_config or MemoryArchiveConfig()
        )
        memory_archive_reflector = ArchiveMemoryReflector(
            self.registry.get(self.provider),
            default_model=self.model,
            provider=self.settings.model_default_provider,
            config=archive_config,
        )

        context_settings = ContextSettings()
        summary_config = self._context_summary_config or ContextSummaryModelConfig(
            enabled=self.context_summary_enabled
        )
        summary_provider = (
            ModelProvider(summary_config.provider)
            if summary_config.provider is not None
            else self.settings.model_default_provider
        )
        context_summarizer = (
            ModelContextSummarizer(
                self.registry.get(summary_provider),
                model_provider=summary_provider,
                model=summary_config.model or self.model,
                max_output_tokens=(
                    context_settings.context_summary_max_output_tokens
                ),
            )
            if summary_config.enabled
            else None
        )

        ace = None
        if self.ace_enabled:
            ace = AceCoordinator(
                self.registry.get(self.provider),
                model=self.model,
                path=self.ace_playbook_path,
                max_strategies=self.ace_max_strategies,
                select_top_k=self.ace_select_top_k,
                irrelevant_streak_limit=self.ace_irrelevant_streak_limit,
                max_output_tokens=self.ace_max_output_tokens,
                bulletpoint_analyzer_enabled=self.ace_bulletpoint_analyzer_enabled,
                bulletpoint_analyzer_threshold=self.ace_bulletpoint_analyzer_threshold,
                max_bullets=self.ace_max_bullets,
                harmful_prune_threshold=self.ace_harmful_prune_threshold,
            )

        # MCP：配置缺失 / 损坏时不阻断启动（CLI 会检查 mcp_error 决定退出码）。
        # mcp_manager: MCPClientManager | None = None
        # mcp_statuses: tuple[Any, ...] = ()
        # mcp_error: str | None = None
        # try:
        #     mcp_settings = await self.mcp_config_store.load()
        #     mcp_manager = MCPClientManager(
        #         mcp_settings.servers,
        #         sandbox_supervisor=sandbox_supervisor,
        #     )
        #     mcp_statuses = await mcp_manager.start(tool_registry)
        # except MCPConfigurationError as exc:
        #     mcp_error = f"{type(exc).__name__}: {exc}"
        #     logger.warning("MCP disabled: %s", mcp_error)
        # # 管理查询只读取当前 Manager 快照，不启动 Server，也不暴露全部 MCP Schema。
        # tool_registry.register(
        #     MCPStatusTool(mcp_manager, configuration_error=mcp_error)
        # )

        context_manager = ContextManager(
            context_settings=context_settings,
            conversation_reducer=(
                ConversationReducer(
                    context_summarizer,
                    keep_recent_conversation_blocks=(
                        context_settings.context_keep_recent_conversation_blocks
                    ),
                    keep_recent_tool_rounds=(
                        context_settings.context_keep_recent_tool_rounds
                    ),
                )
                if context_summarizer is not None
                else None
            ),
        )

        runtime = AgentRuntime(
            handoff_store=handoff_store,
            workspace_root=self.workspace_root,
            model_registry = self.registry,
            provider=self.provider,
            model=self.model,

            tool_registry = tool_registry,
           
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            max_tool_rounds=self.max_tool_rounds,
            max_output_tokens=self.max_output_tokens,
            
            context_manager=context_manager,
            plan_context_provider=PlanContextProvider(plan_store),
            plan_runner=PlanRunner(plan_store),
            checkpoint_store=checkpoint_store,
            memory_manager=memory_manager,
            memory_auto_search_enabled=self.memory_auto_search_enabled,
            memory_reflector=memory_reflector,
            memory_archive_reflector=memory_archive_reflector,
            ace=ace,
            skill_store=skill_store,
            skill_context_provider=skill_context_provider,

            approval_gate=self.approval_gate,
            policy_engine=policy_engine,
            rule_store=rule_store,
            tool_output_recorder=evidence_recorder,
        
            run_budget_config=self._run_budget_config,
        )


        run_store = RunStore(database)
        run_manager = RunManager(
            run_store,
            checkpoint_store,
            runtime
        )
        reconciled_runs = await run_manager.initialize()

        conversation_service = ConversationService(
            conversation_store,
            run_manager,
            trace_store,
            summary_store=summary_store,
        )

      
        skill_candidate_store, skill_improving = await self._build_skill_improving(
            run_store=run_store,
            trace_store=trace_store,
            skill_store=skill_store,
            conversation_store=conversation_store,
        )


        self.conversation_store = conversation_store
        self.summary_store = summary_store
        self.evidence_store = evidence_store
        self.trace_store = trace_store
        self.checkpoint_store = checkpoint_store
        self.rule_store = rule_store
        self.policy_engine = policy_engine
        # self.approval_store = approval_store
        # self.approval_gate = approval_gate
        # self.desktop_approval_gate = (
        #     approval_gate if isinstance(approval_gate, DesktopApprovalGate) else None
        # )
        # self.artifact_store = artifact_store
        # self.artifact_service = artifact_service
        self.tool_registry = tool_registry
        self.plan_store = plan_store
        self.memory_manager = memory_manager
        self.skill_store = skill_store
        self.skill_context_provider = skill_context_provider
        self.skill_candidate_store = skill_candidate_store
        self.skill_improving = skill_improving
        self.memory_reflector = memory_reflector
        self.memory_archive_reflector = memory_archive_reflector
        self.ace = ace
        self.memory_reflection_enabled = reflection_config.enabled
        self.memory_archive_enabled = archive_config.enabled
        self.context_summarizer = context_summarizer
        self.context_manager = context_manager
        self.active_model_roles = {
            "main": {
                "enabled": True,
                "provider": self.provider,
                "model": self.model,
            },
            "summary": {
                "enabled": summary_config.enabled,
                "provider": summary_config.provider or self.provider,
                "model": summary_config.model or self.model,
            },
            "reflection": {
                "enabled": reflection_config.enabled,
                "provider": memory_reflector.provider_hint,
                "model": memory_reflector.model_hint,
            },
            "archive": {
                "enabled": archive_config.enabled,
                "provider": memory_archive_reflector.provider_hint,
                "model": memory_archive_reflector.model_hint,
            },
        }
        # self.mcp_manager = mcp_manager
        # self.mcp_statuses = mcp_statuses
        # self.mcp_error = mcp_error
    
        self.runtime = runtime
        self.run_store = run_store
        self.run_manager = run_manager
        self.conversation_service = conversation_service
        # self.automation_store = automation_store
        # self.automation_scheduler = automation_scheduler
        self.reconciled_runs = reconciled_runs

        self._started = True

    async def _build_skill_improving(
        self,
        *,
        run_store: RunStore,
        trace_store: TraceStore,
        skill_store: SkillStore,
        conversation_store: ConversationStore,
    ) ->tuple[SkillCandidateStore, SkillImproveService]:
        """装配 Skill 自进化管线（CLUSTER → DISTILL → Candidate）。

        默认关闭（``SkillImprovingSettings.enabled=False``）；开启后才创建
        miner/distiller，否则传入 None 使服务降级为 no-op。
        """
        settings = self._skill_improving_settings or SkillImprovingSettings()
        candidate_store = SkillCandidateStore(
            Path(settings.data_dir) / "candidates.db",
            skill_project_dir=skill_store.project_dir,
        )
        await candidate_store.initialize()

        provider = settings.provider or self.provider
        model = settings.model or self.model
        miner = None
        distiller = None
        if settings.enabled:
            adapter = self.registry.get(provider)
            miner = ModelPatternMiner(
                adapter,
                model=model,
                temperature=settings.temperature,
                max_output_tokens=settings.max_output_tokens,
                timeout_seconds=settings.timeout_seconds,
                max_attempts=settings.max_attempts,
            )
            distiller_class = (
                ModelMultiTeacherDistiller
                if settings.improve_method == "multi_teacher"
                else ModelProcedureDistiller
            )
            distiller_kwargs = {}
            if settings.improve_method == "multi_teacher":
                distiller_kwargs["analysis_dir"] = Path(settings.data_dir) / "teacher-analyses"
            distiller = distiller_class(
                adapter,
                model=model,
                temperature=settings.temperature,
                max_output_tokens=settings.max_output_tokens,
                timeout_seconds=settings.timeout_seconds,
                max_attempts=settings.max_attempts,
                **distiller_kwargs,
            )

        service = SkillImproveService(
            run_store=run_store,
            trace_store=trace_store,
            skill_store=skill_store,
            candidate_store=candidate_store,
            registry=self.registry,
            settings=settings,
            conversation_store=conversation_store,
            miner=miner,
            distiller=distiller,
            selector=DefaultEventSelector(),
            evidence_builder=TraceEvidenceBuilder(),
        )
        return candidate_store, service

    async def close(self) -> None:
        """关闭模型注册表等外部资源。"""
        if self.runtime is not None:
            await self.runtime.flush_background_tasks()
        await self.registry.close()
