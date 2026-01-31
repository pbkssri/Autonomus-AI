"""
Main Agent Core - Central Orchestrator for Autonomous AI Agent
Integrates all modules and provides unified interface for AI operations
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from .config_manager import ConfigurationManager, AgentConfig
from .api_providers import APIProviderFactory, APIResponse, AIProvider, BaseAPIProvider
from .task_orchestrator import TaskOrchestrator, Task, TaskResult, TaskStatus, TaskPriority
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class AgentCore:
    """Main agent core that orchestrates all operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize agent core
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigurationManager(config_path)
        self.memory = MemoryManager()
        self.task_orchestrator: Optional[TaskOrchestrator] = None
        
        self._providers: Dict[AIProvider, BaseAPIProvider] = {}
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt: str = ""
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default system prompt
        self._init_system_prompt()
    
    def _init_system_prompt(self) -> None:
        """Initialize default system prompt"""
        self._system_prompt = """You are an autonomous AI assistant powered by Gemini Pro and ChatGPT Pro.
Your capabilities include:
- Task planning and execution
- Information research and synthesis
- File operations and code development
- Problem solving and analysis
- Conversation and reasoning

Guidelines:
1. Think step-by-step before responding
2. Ask clarifying questions when needed
3. Provide accurate, helpful information
4. Be transparent about limitations
5. Learn from interactions and remember important context

You operate autonomously but can always defer to user input when appropriate."""
    
    def initialize(self) -> bool:
        """Initialize agent with configured API keys
        
        Returns:
            True if initialization successful
        """
        try:
            # Get API keys from config
            keys = self.config.get_api_keys()
            
            # Initialize providers
            if keys['gemini']:
                provider = APIProviderFactory.create_provider(
                    AIProvider.GEMINI,
                    keys['gemini'],
                    self.config.config.max_tokens,
                    self.config.config.temperature
                )
                self._providers[AIProvider.GEMINI] = provider
                logger.info("Gemini provider initialized")
            
            if keys['openai']:
                provider = APIProviderFactory.create_provider(
                    AIProvider.OPENAI,
                    keys['openai'],
                    self.config.config.max_tokens,
                    self.config.config.temperature
                )
                self._providers[AIProvider.OPENAI] = provider
                logger.info("OpenAI provider initialized")
            
            # Initialize task orchestrator
            self.task_orchestrator = TaskOrchestrator(
                self,
                self.config.config.max_concurrent_tasks
            )
            
            logger.info("Agent core initialized successfully")
            return self.is_ready()
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if agent is ready to operate
        
        Returns:
            True if at least one provider is configured
        """
        return bool(self._providers)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers
        
        Returns:
            List of provider names
        """
        return [p.provider_name for p in self._providers.values() if p.is_configured]
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt for AI responses
        
        Args:
            prompt: New system prompt
        """
        self._system_prompt = prompt
        logger.info("System prompt updated")
    
    def add_conversation_message(self, role: str, content: str) -> None:
        """Add message to conversation history
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        self._conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history length
        max_history = 50
        if len(self._conversation_history) > max_history:
            self._conversation_history = self._conversation_history[-max_history:]
    
    def clear_conversation(self) -> None:
        """Clear conversation history"""
        self._conversation_history = []
    
    async def execute_prompt(self, system_prompt: str, user_prompt: str,
                           provider: Optional[AIProvider] = None) -> str:
        """Execute a prompt using AI
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            provider: Specific provider to use (auto-select if not specified)
            
        Returns:
            AI response text
        """
        # Build messages
        messages = [{"role": "user", "content": user_prompt}]
        
        # Select provider
        if provider and provider in self._providers:
            selected_provider = self._providers[provider]
        elif len(self._providers) == 1:
            selected_provider = next(iter(self._providers.values()))
        elif AIProvider.OPENAI in self._providers:
            selected_provider = self._providers[AIProvider.OPENAI]
        elif AIProvider.GEMINI in self._providers:
            selected_provider = self._providers[AIProvider.GEMINI]
        else:
            raise ValueError("No AI provider configured")
        
        # Add to conversation history
        self.add_conversation_message("user", user_prompt)
        
        # Execute
        response = await selected_provider.chat_completion(
            messages,
            system_prompt or self._system_prompt
        )
        
        if response.success:
            self.add_conversation_message("assistant", response.content)
            
            # Save to memory if important
            if self.config.config.memory_enabled:
                await self.memory.remember(
                    content=f"User: {user_prompt}\nAssistant: {response.content}",
                    category="conversation",
                    importance=0.5,
                    metadata={"provider": selected_provider.provider_name}
                )
            
            return response.content
        else:
            error_msg = response.error or "Unknown error"
            logger.error(f"Prompt execution failed: {error_msg}")
            raise RuntimeError(f"AI request failed: {error_msg}")
    
    async def stream_prompt(self, system_prompt: str, user_prompt: str,
                           provider: Optional[AIProvider] = None) -> AsyncGenerator[str, None]:
        """Stream AI response
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            provider: Specific provider to use
            
        Yields:
            Response text chunks
        """
        messages = [{"role": "user", "content": user_prompt}]
        
        if provider and provider in self._providers:
            selected_provider = self._providers[provider]
        elif AIProvider.OPENAI in self._providers:
            selected_provider = self._providers[AIProvider.OPENAI]
        elif AIProvider.GEMINI in self._providers:
            selected_provider = self._providers[AIProvider.GEMINI]
        else:
            raise ValueError("No AI provider configured")
        
        async for chunk in selected_provider.stream_completion(messages, system_prompt):
            yield chunk
    
    async def chat(self, message: str) -> str:
        """Chat with the agent
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        return await self.execute_prompt(self._system_prompt, message)
    
    async def analyze(self, content: str, analysis_type: str = "general") -> str:
        """Analyze content
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis
            
        Returns:
            Analysis result
        """
        analysis_prompts = {
            "general": "Analyze the following content and provide key insights:\n{content}",
            "code": "Review the following code and provide feedback on quality, potential issues, and improvements:\n{content}",
            "document": "Summarize the following document:\n{content}",
            "data": "Analyze the following data and identify patterns:\n{content}",
"sentiment": "Analyze the sentiment of the following text:\n{content}"
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        return await self.execute_prompt(
            f"You are an expert analyst. {analysis_type.capitalize()} analysis mode.",
            prompt.format(content=content)
        )
    
    async def research(self, topic: str, depth: str = "basic") -> str:
        """Research a topic
        
        Args:
            topic: Topic to research
            depth: Research depth (basic/intermediate/comprehensive)
            
        Returns:
            Research results
        """
        depth_guidance = {
            "basic": "Provide a brief overview with key points.",
            "intermediate": "Provide a detailed analysis with examples.",
            "comprehensive": "Provide an exhaustive analysis with history, current state, and future implications."
        }
        
        return await self.execute_prompt(
            """You are a research assistant. Provide accurate, well-sourced information.
Include different perspectives where applicable.""",
            f"Research the following topic ({depth} level). {depth_guidance.get(depth, '')}\n\nTopic: {topic}"
        )
    
    async def generate_code(self, requirements: str, language: str = "python") -> str:
        """Generate code
        
        Args:
            requirements: Code requirements
            language: Programming language
            
        Returns:
            Generated code
        """
        return await self.execute_prompt(
            f"""You are an expert {language} developer. Write clean, well-documented code.
Follow best practices and include error handling.
Provide only the code, no explanations unless requested.""",
            f"Write {language} code for the following requirements:\n{requirements}"
        )
    
    async def plan_task(self, goal: str, context: Optional[Dict] = None) -> List[Task]:
        """Plan a complex task
        
        Args:
            goal: Goal to achieve
            context: Additional context
            
        Returns:
            List of planned tasks
        """
        if not self.task_orchestrator:
            raise RuntimeError("Task orchestrator not initialized")
        
        tasks = await self.task_orchestrator.planner.create_plan(goal, context)
        
        # Store in memory
        if self.config.config.memory_enabled:
            await self.memory.remember(
                content=f"Task plan: {goal}\nTasks: {json.dumps([t.to_dict() for t in tasks])}",
                category="task",
                importance=0.7,
                metadata={"goal": goal}
            )
        
        return tasks
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        if not self.task_orchestrator:
            raise RuntimeError("Task orchestrator not initialized")
        
        return await self.task_orchestrator.execute_task(task)
    
    async def submit_and_execute(self, task_description: str, 
                                auto_plan: bool = True) -> TaskResult:
        """Submit and execute a task
        
        Args:
            task_description: Description of the task
            auto_plan: Whether to auto-plan
            
        Returns:
            Task execution result
        """
        task = Task(
            id="",
            name=task_description[:50] + "..." if len(task_description) > 50 else task_description,
            description=task_description,
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        task_id = await self.task_orchestrator.submit_task(task, auto_plan)
        
        # Wait for completion (simplified - in production would be async)
        while True:
            status = self.task_orchestrator.get_task_status(task_id)
            if not status:
                break
            if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return TaskResult(
                    success=status.status == TaskStatus.COMPLETED,
                    output=status.output_data,
                    error=None if status.status == TaskStatus.COMPLETED else "Task failed"
                )
            await asyncio.sleep(0.1)
        
        return TaskResult(success=False, output=None, error="Task not found")
    
    def get_status(self) -> Dict:
        """Get agent status
        
        Returns:
            Status dictionary
        """
        return {
            "ready": self.is_ready(),
            "providers": self.get_available_providers(),
            "memory_entries": self.memory.get_stats(),
            "conversation_history_length": len(self._conversation_history),
            "config": {
                "max_tokens": self.config.config.max_tokens,
                "temperature": self.config.config.temperature,
                "autonomous_planning": self.config.config.autonomous_planning
            }
        }
    
    async def validate_keys(self) -> Dict[str, bool]:
        """Validate API keys
        
        Returns:
            Dictionary of provider to validation status
        """
        results = {}
        for provider_type, provider in self._providers.items():
            results[provider_type.value] = provider.validate_connection()
        return results
    
    def shutdown(self) -> None:
        """Shutdown agent and cleanup resources"""
        if self.task_orchestrator:
            self.task_orchestrator.stop()
        
        self._executor.shutdown(wait=False)
        logger.info("Agent core shutdown complete")
