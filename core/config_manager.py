"""
Configuration Manager for Autonomous AI Agent
Handles loading, saving, and managing configuration settings
"""

import os
import json
import configparser
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration data class for agent settings"""
    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    
    # Agent Settings
    max_tokens: int = 4096
    temperature: float = 0.7
    auto_save: bool = True
    workspace_dir: str = "./workspace"
    logging_enabled: bool = True
    log_level: str = "INFO"
    
    # Task Settings
    max_concurrent_tasks: int = 5
    task_timeout: int = 300
    auto_retry: bool = True
    max_retries: int = 3
    
    # Features
    web_search_enabled: bool = True
    file_operations_enabled: bool = True
    code_execution_enabled: bool = True
    memory_enabled: bool = True
    autonomous_planning: bool = True
    
    # GUI Settings
    window_width: int = 1200
    window_height: int = 800
    theme: str = "dark"
    always_on_top: bool = False
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            return False
        if not isinstance(self.temperature, float) or not (0.0 <= self.temperature <= 1.0):
            return False
        if not isinstance(self.max_concurrent_tasks, int) or self.max_concurrent_tasks < 1:
            return False
        if self.gemini_api_key and not self.gemini_api_key.startswith("AIza"):
            logger.warning("Gemini API key format may be incorrect")
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            logger.warning("OpenAI API key format may be incorrect")
        return True


class ConfigurationManager:
    """Manages application configuration with file persistence"""
    
    DEFAULT_CONFIG = AgentConfig()
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (default: config/config.ini)
        """
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "config.ini"
        
        self.config_path = Path(config_path)
        self.config = AgentConfig()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        if not self.config_path.exists():
            logger.info(f"Config file not found, using defaults: {self.config_path}")
            self._save_config()
            return
        
        try:
            config = configparser.ConfigParser()
            config.read(self.config_path, encoding='utf-8')
            
            # Load API keys
            self.config.gemini_api_key = config.get('API_KEYS', 'GEMINI_API_KEY', fallback="")
            self.config.openai_api_key = config.get('API_KEYS', 'OPENAI_API_KEY', fallback="")
            
            # Load agent settings
            self.config.max_tokens = config.getint('AGENT_SETTINGS', 'MAX_TOKENS', fallback=self.DEFAULT_CONFIG.max_tokens)
            self.config.temperature = config.getfloat('AGENT_SETTINGS', 'TEMPERATURE', fallback=self.DEFAULT_CONFIG.temperature)
            self.config.auto_save = config.getboolean('AGENT_SETTINGS', 'AUTO_SAVE', fallback=self.DEFAULT_CONFIG.auto_save)
            self.config.workspace_dir = config.get('AGENT_SETTINGS', 'WORKSPACE_DIR', fallback=self.DEFAULT_CONFIG.workspace_dir)
            self.config.logging_enabled = config.getboolean('AGENT_SETTINGS', 'LOGGING_ENABLED', fallback=True)
            self.config.log_level = config.get('AGENT_SETTINGS', 'LOG_LEVEL', fallback="INFO")
            
            # Load task settings
            self.config.max_concurrent_tasks = config.getint('TASK_SETTINGS', 'MAX_CONCURRENT_TASKS', fallback=5)
            self.config.task_timeout = config.getint('TASK_SETTINGS', 'TASK_TIMEOUT', fallback=300)
            self.config.auto_retry = config.getboolean('TASK_SETTINGS', 'AUTO_RETRY', fallback=True)
            self.config.max_retries = config.getint('TASK_SETTINGS', 'MAX_RETRIES', fallback=3)
            
            # Load features
            self.config.web_search_enabled = config.getboolean('FEATURES', 'WEB_SEARCH_ENABLED', fallback=True)
            self.config.file_operations_enabled = config.getboolean('FEATURES', 'FILE_OPERATIONS_ENABLED', fallback=True)
            self.config.code_execution_enabled = config.getboolean('FEATURES', 'CODE_EXECUTION_ENABLED', fallback=True)
            self.config.memory_enabled = config.getboolean('FEATURES', 'MEMORY_ENABLED', fallback=True)
            self.config.autonomous_planning = config.getboolean('FEATURES', 'AUTONOMOUS_PLANNING', fallback=True)
            
            # Load GUI settings
            self.config.window_width = config.getint('GUI_SETTINGS', 'WINDOW_WIDTH', fallback=1200)
            self.config.window_height = config.getint('GUI_SETTINGS', 'WINDOW_HEIGHT', fallback=800)
            self.config.theme = config.get('GUI_SETTINGS', 'THEME', fallback="dark")
            self.config.always_on_top = config.getboolean('GUI_SETTINGS', 'ALWAYS_ON_TOP', fallback=False)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = AgentConfig()
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = configparser.ConfigParser()
            
            # Save API keys
            config['API_KEYS'] = {
                'GEMINI_API_KEY': self.config.gemini_api_key,
                'OPENAI_API_KEY': self.config.openai_api_key
            }
            
            # Save agent settings
            config['AGENT_SETTINGS'] = {
                'MAX_TOKENS': str(self.config.max_tokens),
                'TEMPERATURE': str(self.config.temperature),
                'AUTO_SAVE': str(self.config.auto_save),
                'WORKSPACE_DIR': self.config.workspace_dir,
                'LOGGING_ENABLED': str(self.config.logging_enabled),
                'LOG_LEVEL': self.config.log_level
            }
            
            # Save task settings
            config['TASK_SETTINGS'] = {
                'MAX_CONCURRENT_TASKS': str(self.config.max_concurrent_tasks),
                'TASK_TIMEOUT': str(self.config.task_timeout),
                'AUTO_RETRY': str(self.config.auto_retry),
                'MAX_RETRIES': str(self.config.max_retries)
            }
            
            # Save features
            config['FEATURES'] = {
                'WEB_SEARCH_ENABLED': str(self.config.web_search_enabled),
                'FILE_OPERATIONS_ENABLED': str(self.config.file_operations_enabled),
                'CODE_EXECUTION_ENABLED': str(self.config.code_execution_enabled),
                'MEMORY_ENABLED': str(self.config.memory_enabled),
                'AUTONOMOUS_PLANNING': str(self.config.autonomous_planning)
            }
            
            # Save GUI settings
            config['GUI_SETTINGS'] = {
                'WINDOW_WIDTH': str(self.config.window_width),
                'WINDOW_HEIGHT': str(self.config.window_height),
                'THEME': self.config.theme,
                'ALWAYS_ON_TOP': str(self.config.always_on_top)
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                config.write(f)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def update_api_keys(self, gemini_key: str = "", openai_key: str = "") -> bool:
        """Update API keys
        
        Args:
            gemini_key: New Gemini API key
            openai_key: New OpenAI API key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.gemini_api_key = gemini_key.strip()
            self.config.openai_api_key = openai_key.strip()
            self._save_config()
            return True
        except Exception as e:
            logger.error(f"Error updating API keys: {e}")
            return False
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get current API keys
        
        Returns:
            Dictionary containing API keys
        """
        return {
            'gemini': self.config.gemini_api_key,
            'openai': self.config.openai_api_key
        }
    
    def is_configured(self) -> bool:
        """Check if agent is properly configured
        
        Returns:
            True if at least one API key is configured
        """
        return bool(self.config.gemini_api_key or self.config.openai_api_key)
    
    def get_config(self) -> AgentConfig:
        """Get full configuration
        
        Returns:
            Current configuration object
        """
        return self.config
    
    def update_settings(self, **kwargs) -> bool:
        """Update configuration settings
        
        Args:
            **kwargs: Configuration settings to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            allowed_fields = {
                'max_tokens', 'temperature', 'auto_save', 'workspace_dir',
                'logging_enabled', 'log_level', 'max_concurrent_tasks',
                'task_timeout', 'auto_retry', 'max_retries',
                'web_search_enabled', 'file_operations_enabled',
                'code_execution_enabled', 'memory_enabled',
                'autonomous_planning', 'window_width', 'window_height',
                'theme', 'always_on_top'
            }
            
            for key, value in kwargs.items():
                if key in allowed_fields:
                    setattr(self.config, key, value)
            
            self._save_config()
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
