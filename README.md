Autonomous AI Agent

A powerful Windows desktop application that acts as an autonomous AI assistant, powered by your Gemini Pro and ChatGPT Pro API accounts.

## Features

- **Autonomous Task Execution**: Create and execute complex tasks with AI-powered planning
- **Multi-Provider Support**: Works with both Google Gemini Pro and OpenAI ChatGPT Pro
- **Web Search & Research**: Built-in web search and content extraction capabilities
- **File Operations**: Safe file reading, writing, and management
- **Code Execution**: Secure Python and JavaScript code execution
- **Persistent Memory**: Agent remembers context and learns over time
- **Modern GUI**: Clean, intuitive interface for easy interaction
- **Task History**: Track and manage all your AI tasks

## Requirements

- Windows 10 or Windows 11
- Python 3.8 or higher (for building from source)
- API keys for at least one AI provider:
  - Google Gemini API: https://aistudio.google.com/
  - OpenAI API (ChatGPT): https://platform.openai.com/api-keys

## Quick Start

### Option 1: Using the Pre-built Executable

1. Navigate to the `dist\AutonomousAI-Agent` folder
2. Double-click `AutonomousAI-Agent.exe` to launch the application
3. Go to **File > Settings** and enter your API keys
4. Start using your autonomous AI agent!

### Option 2: Building from Source

1. Install Python 3.8+ from https://python.org
2. Clone or download this repository
3. Run `build.bat` to build the Windows executable
4. Or run `python main.py` directly for development

### Option 3: Running from Source

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

### GUI Mode

1. Launch the application
2. Configure your API keys in Settings
3. Select a mode:
   - **Chat**: General conversation
   - **Task**: Execute autonomous tasks
   - **Research**: Deep research on topics
   - **Code**: Generate code snippets
4. Enter your task/message and click Execute

### Command Line Mode

```bash
# Execute a task from command line
python main.py "Research the latest developments in AI"

# Or use the built executable
AutonomousAI-Agent.exe "Research the latest developments in AI"
```

## Configuration

### API Keys

Get your API keys from:
- **Gemini**: https://aistudio.google.com/apikey
- **OpenAI**: https://platform.openai.com/api-keys

### Settings

Configure the following in Settings:
- **Max Tokens**: Response length (default: 4096)
- **Temperature**: Creativity level (0.0-1.0, default: 0.7)
- **Max Concurrent Tasks**: Task parallelization (default: 5)
- **Task Timeout**: Maximum task duration in seconds (default: 300)
- **Features**: Enable/disable web search, file operations, code execution, memory

## Architecture

```
autonomous_agent/
├── core/                   # Core agent modules
│   ├── agent_core.py       # Main orchestrator
│   ├── config_manager.py   # Configuration management
│   ├── api_providers.py    # AI provider integration
│   ├── task_orchestrator.py # Task planning & execution
│   └── memory_manager.py   # Persistent memory
├── gui/                    # GUI components
│   ├── main_window.py      # Main application window
│   └── settings_dialog.py  # Settings configuration
├── tools/                  # External tools
│   ├── file_operations.py  # File management
│   ├── web_search.py       # Web search & extraction
│   └── code_execution.py   # Safe code execution
├── config/                 # Configuration files
│   └── config.ini          # User configuration
├── data/                   # Data storage
│   └── memory.db           # SQLite memory database
└── workspace/              # User workspace
```

## Security

- **API Keys**: Stored locally in `config/config.ini` (not transmitted externally)
- **Code Execution**: Sandboxed with dangerous operation detection
- **File Operations**: Restricted to workspace directory by default
- **Memory**: Local SQLite database, no cloud sync

## Troubleshooting

### "API Key not configured"
- Go to File > Settings and enter your API keys
- Keys must start with "AIza" (Gemini) or "sk-" (OpenAI)

### "Connection failed"
- Check your internet connection
- Verify API keys are correct
- Ensure API access is enabled for your account

### Application won't start
- Check if antivirus is blocking the executable
- Try running as administrator
- Check logs in `logs/agent.log`

## Building the Executable

```bash
# Install build dependencies
pip install pyinstaller

# Build
pyinstaller autonomous_agent.spec

# Or use the build script
build.bat
```

The executable will be in `dist/AutonomousAI-Agent/`

## License

MIT License - Use freely for personal and commercial purposes.

## Support

For issues and feature requests, please check the GitHub repository.

---

**Note**: This agent uses your existing Gemini Pro and ChatGPT Pro API accounts. You are responsible for any API usage costs incurred through your accounts.
