"""
Task Orchestrator for Autonomous AI Agent
Manages task execution, planning, and coordination
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import logging
import uuid
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = auto()
    PLANNING = auto()
    IN_PROGRESS = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    PAUSED = auto()


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of task execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


@dataclass
class Task:
    """Task representation"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    subtasks: List['Task'] = field(default_factory=list)
    parent_task: Optional['Task'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    executor_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "dependencies": self.dependencies
        }


class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @property
    @abstractmethod
    def executor_name(self) -> str:
        """Get executor name"""
        pass
    
    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        pass
    
    @abstractmethod
    def can_execute(self, task: Task) -> bool:
        """Check if executor can handle this task
        
        Args:
            task: Task to check

        Returns:
            True if executor can handle the task
        """
        pass


class TaskPlanner:
    """AI-powered task planning and decomposition"""
    
    def __init__(self, agent_core: 'AgentCore'):
        """Initialize task planner
        
        Args:
            agent_core: Reference to main agent core
        """
        self.agent = agent_core
    
    async def create_plan(self, goal: str, context: Optional[Dict] = None) -> List[Task]:
        """Create a plan to achieve a goal
        
        Args:
            goal: High-level goal to accomplish
            context: Additional context information
            
        Returns:
            List of planned tasks
        """
        system_prompt = """You are a task planning assistant. Your role is to break down complex goals into actionable tasks.
        
Rules:
1. Create clear, atomic tasks that can be executed independently
2. Consider dependencies between tasks
3. Assign appropriate priorities based on task importance
4. Each task should have a clear description and expected outcome
5. Return your response as a valid JSON array of tasks

Task format:
{
    "name": "Descriptive task name",
    "description": "Detailed description of what needs to be done",
    "priority": "HIGH|MEDIUM|LOW",
    "estimated_duration": "Time estimate in minutes",
    "dependencies": ["task_ids that must complete first"]
}

Return only the JSON array, no additional text."""

        user_message = f"Goal: {goal}\n\nContext: {json.dumps(context) if context else 'No additional context'}"

        try:
            response = await self.agent.execute_prompt(system_prompt, user_message)
            
            # Parse the response
            tasks_data = json.loads(response)
            
            tasks = []
            for i, task_data in enumerate(tasks_data):
                task = Task(
                    id=str(uuid.uuid4()),
                    name=task_data.get("name", f"Task {i+1}"),
                    description=task_data.get("description", ""),
                    priority=TaskPriority[task_data.get("priority", "MEDIUM")],
                    status=TaskStatus.PENDING,
                    created_at=datetime.now(),
                    metadata={
                        "estimated_duration": task_data.get("estimated_duration", 0),
                        "original_goal": goal
                    }
                )
                tasks.append(task)
            
            return tasks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}")
            # Return a simple fallback task
            return [Task(
                id=str(uuid.uuid4()),
                name="Complete Goal",
                description=goal,
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )]
        except Exception as e:
            logger.error(f"Planning error: {e}")
            raise
    
    async def analyze_and_decompose(self, task: Task) -> List[Task]:
        """Analyze a task and decompose into subtasks if needed
        
        Args:
            task: Task to analyze
            
        Returns:
            List of subtasks (empty if task is atomic)
        """
        system_prompt = """You are a task decomposition assistant. Analyze the given task and determine if it can be broken down into smaller subtasks.

If the task is complex and can be decomposed:
- Break it into 2-5 smaller, manageable subtasks
- Each subtask should be atomic and independently executable
- Return a JSON array of subtasks

If the task is already atomic/simple:
- Return an empty array

Subtask format:
{
    "name": "Descriptive subtask name",
    "description": "What this subtask accomplishes",
    "priority": "HIGH|MEDIUM|LOW"
}

Return only the JSON array, no additional text."""

        user_message = f"Task: {task.name}\nDescription: {task.description}\nCurrent status: {task.status.name}"

        try:
            response = await self.agent.execute_prompt(system_prompt, user_message)
            subtasks_data = json.loads(response)
            
            if not subtasks_data:
                return []
            
            subtasks = []
            for i, subtask_data in enumerate(subtasks_data):
                subtask = Task(
                    id=str(uuid.uuid4()),
                    name=subtask_data.get("name", f"Subtask {i+1}"),
                    description=subtask_data.get("description", ""),
                    priority=TaskPriority[subtask_data.get("priority", "MEDIUM")],
                    status=TaskStatus.PENDING,
                    created_at=datetime.now(),
                    parent_task=task
                )
                subtasks.append(subtask)
                task.subtasks.append(subtask)
            
            return subtasks
            
        except json.JSONDecodeError:
            return []
        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            return []


class TaskOrchestrator:
    """Main task orchestration system"""
    
    def __init__(self, agent_core: 'AgentCore', max_concurrent: int = 5):
        """Initialize task orchestrator
        
        Args:
            agent_core: Reference to main agent core
            max_concurrent: Maximum concurrent tasks
        """
        self.agent = agent_core
        self.planner = TaskPlanner(agent_core)
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.executors: Dict[str, TaskExecutor] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.max_concurrent = max_concurrent
        self.executor_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self._running = False
        
    def register_executor(self, executor: TaskExecutor) -> None:
        """Register a task executor
        
        Args:
            executor: Executor to register
        """
        self.executors[executor.executor_name] = executor
        logger.info(f"Registered executor: {executor.executor_name}")
    
    async def submit_task(self, task: Task, auto_plan: bool = True) -> str:
        """Submit a task for execution
        
        Args:
            task: Task to submit
            auto_plan: Whether to automatically plan complex tasks
            
        Returns:
            Task ID
        """
        if auto_plan and self.agent.config.autonomous_planning:
            # Analyze and decompose if needed
            subtasks = await self.planner.analyze_and_decompose(task)
            if subtasks:
                # Add subtasks to queue
                for subtask in subtasks:
                    subtask.dependencies = [task.id]
                    await self.task_queue.put(subtask)
                task.subtasks = subtasks
        
        await self.task_queue.put(task)
        logger.info(f"Task submitted: {task.name} (ID: {task.id})")
        return task.id
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        import time
        start_time = time.time()
        
        task.started_at = datetime.now()
        task.status = TaskStatus.IN_PROGRESS
        
        # Find suitable executor
        executor = None
        for ex in self.executors.values():
            if ex.can_execute(task):
                executor = ex
                break
        
        if not executor:
            # Use default execution through agent core
            return await self._default_execution(task)
        
        try:
            result = await executor.execute(task)
            
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.output_data = result.output
            task.completed_at = datetime.now()
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            logger.info(f"Task completed: {task.name} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            error_msg = str(e)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.warning(f"Task {task.name} failed, retry {task.retry_count}/{task.max_retries}")
                return await self.execute_task(task)
            
            logger.error(f"Task failed: {task.name} - {error_msg}")
            
            return TaskResult(
                success=False,
                output=None,
                error=error_msg,
                execution_time=time.time() - start_time
            )
    
    async def _default_execution(self, task: Task) -> TaskResult:
        """Default task execution through AI
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        try:
            response = await self.agent.execute_prompt(
                f"Execute the following task: {task.description}",
                f"Task context: {json.dumps(task.input_data)}"
            )
            
            return TaskResult(
                success=True,
                output=response,
                metadata={"executor": "default"}
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def run(self) -> None:
        """Main task processing loop"""
        self._running = True
        
        while self._running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                result = await self.execute_task(task)
                self.completed_tasks.append(task)
                
                # Process subtasks if any
                for subtask in task.subtasks:
                    if subtask.status == TaskStatus.PENDING:
                        subtask.dependencies = [t.id for t in task.subtasks if t.id != subtask.id]
                        await self.task_queue.put(subtask)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error: {e}")
    
    def stop(self) -> None:
        """Stop the orchestrator"""
        self._running = False
        self.executor_pool.shutdown(wait=False)
        logger.info("Task orchestrator stopped")
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get status of a task
        
        Args:
            task_id: Task ID to look up
            
        Returns:
            Task if found, None otherwise
        """
        return self.active_tasks.get(task_id)
    
    def list_active_tasks(self) -> List[Dict]:
        """List all active tasks
        
        Returns:
            List of task summaries
        """
        return [t.to_dict() for t in self.active_tasks.values()]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        task = self.active_tasks.get(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.PLANNING]:
            task.status = TaskStatus.CANCELLED
            return True
        return False
