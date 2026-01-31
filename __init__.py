"""
Autonomous AI Agent - Main Package
An intelligent agent powered by Gemini Pro and ChatGPT Pro
"""

__version__ = "1.0.0"
__author__ = "Matrix Agent"

from .core.agent_core import AgentCore
from .core.config_manager import ConfigurationManager
from .core.memory_manager import MemoryManager

__all__ = ['AgentCore', 'ConfigurationManager', 'MemoryManager']
