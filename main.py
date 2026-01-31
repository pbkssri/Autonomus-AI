#!/usr/bin/env python3
"""
Autonomous AI Agent - Main Entry Point
A Windows executable AI assistant powered by Gemini Pro and ChatGPT Pro
"""

import sys
import os
import logging
from pathlib import Path

# Add the project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))


def setup_logging(log_file: str = "agent.log") -> logging.Logger:
    """Setup logging configuration
    
    Args:
        log_file: Log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Autonomous AI Agent starting...")
    return logger


def check_dependencies() -> bool:
    """Check if all required dependencies are installed
    
    Returns:
        True if all dependencies are available
    """
    required_packages = [
        ('aiohttp', 'aiohttp'),
        ('tkinter', 'tkinter')  # Built-in
    ]
    
    missing = []
    for package, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return False
    
    return True


def initialize_agent() -> tuple:
    """Initialize the agent core and configuration
    
    Returns:
        Tuple of (agent_core, config_manager)
    """
    from core.config_manager import ConfigurationManager
    from core.agent_core import AgentCore
    
    # Initialize configuration
    config_path = project_dir / "config" / "config.ini"
    config_manager = ConfigurationManager(str(config_path))
    
    # Initialize agent core
    agent = AgentCore(str(config_path))
    
    return agent, config_manager


def run_gui(agent, config_manager) -> None:
    """Run the GUI application
    
    Args:
        agent: Agent core instance
        config_manager: Configuration manager instance
    """
    from gui.main_window import AgentGUI
    
    gui = AgentGUI(agent, config_manager)
    gui.run()


def run_cli(agent, task: str) -> None:
    """Run the agent in CLI mode
    
    Args:
        agent: Agent core instance
        task: Task to execute
    """
    import asyncio
    
    async def execute():
        result = await agent.submit_and_execute(task)
        print("\nResult:")
        print(result.output if result.success else f"Error: {result.error}")
    
    asyncio.run(execute())


def main():
    """Main entry point"""
    # Setup logging
    logger = setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Initialize agent
        agent, config_manager = initialize_agent()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # CLI mode
            task = " ".join(sys.argv[1:])
            print(f"Executing task: {task}")
            run_cli(agent, task)
        else:
            # GUI mode
            logger.info("Starting GUI mode")
            run_gui(agent, config_manager)
        
        # Cleanup
        agent.shutdown()
        logger.info("Agent shutdown complete")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
