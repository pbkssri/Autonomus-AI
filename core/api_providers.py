"""
API Provider Factory and Manager
Handles creation and management of different AI API providers
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class APIResponse:
    """Standard API response structure"""
    success: bool
    content: str
    error: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAPIProvider(ABC):
    """Abstract base class for AI API providers"""
    
    def __init__(self, api_key: str, max_tokens: int = 4096, temperature: float = 0.7):
        """Initialize API provider
        
        Args:
            api_key: API key for authentication
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
        """
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._is_configured = False
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name"""
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             system_prompt: Optional[str] = None) -> APIResponse:
        """Generate chat completion
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            APIResponse with generated content
        """
        pass
    
    @abstractmethod
    async def stream_completion(self, messages: List[Dict[str, str]],
                               system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream completion response
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            
        Yields:
            Generated text chunks
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate API connection
        
        Returns:
            True if connection is valid
        """
        pass
    
    @property
    def is_configured(self) -> bool:
        """Check if provider is configured"""
        return self._is_configured


class GeminiProvider(BaseAPIProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: str, max_tokens: int = 4096, temperature: float = 0.7):
        super().__init__(api_key, max_tokens, temperature)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-1.5-pro"  # Using Gemini Pro model
    
    @property
    def provider_name(self) -> str:
        return "Google Gemini"
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             system_prompt: Optional[str] = None) -> APIResponse:
        """Generate chat completion using Gemini API"""
        import aiohttp
        
        try:
            # Build the content from messages
            if system_prompt:
                content = f"System: {system_prompt}\n\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in messages
                )
            else:
                content = "\n".join(
f"{msg['role']}: {msg['content']}" for msg in messages
                )
            
            url = f"{self.base_url}/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            payload = {
                "contents": [{
                    "parts": [{"text": content}],
                    "role": "user"
                }],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract generated text
                        text = ""
                        if "candidates" in data and len(data["candidates"]) > 0:
                            candidate = data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                text = "".join(
                                    part.get("text", "") 
                                    for part in candidate["content"]["parts"]
                                )
                        
                        # Extract usage information
                        usage = None
                        if "usageMetadata" in data:
                            usage = {
                                "prompt_tokens": data["usageMetadata"].get("promptTokenCount", 0),
                                "completion_tokens": data["usageMetadata"].get("candidatesTokenCount", 0),
                                "total_tokens": data["usageMetadata"].get("totalTokenCount", 0)
                            }
                        
                        self._is_configured = True
                        
                        return APIResponse(
                            success=True,
                            content=text,
                            model=self.model,
                            usage=usage
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {response.status} - {error_text}")
                        return APIResponse(
                            success=False,
                            content="",
                            error=f"API error: {response.status}"
                        )
                        
        except Exception as e:
            logger.error(f"Gemini API exception: {e}")
            return APIResponse(
                success=False,
                content="",
                error=str(e)
            )
    
    async def stream_completion(self, messages: List[Dict[str, str]],
                               system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream completion from Gemini API"""
        # Note: Gemini API doesn't support streaming in the same way as OpenAI
        # This is a simplified implementation
        response = await self.chat_completion(messages, system_prompt)
        if response.success:
            yield response.content
        elif response.error:
            yield f"Error: {response.error}"
    
    def validate_connection(self) -> bool:
        """Validate Gemini API connection"""
        import aiohttp
        
        try:
            url = f"{self.base_url}/models/{self.model}?key={self.api_key}"
            
            import asyncio
            response = asyncio.run(self._async_validate(url))
            
            if response.status == 200:
                self._is_configured = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"Gemini validation error: {e}")
            return False
    
    async def _async_validate(self, url: str):
        """Async helper for validation"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            return await session.get(url)


class OpenAIProvider(BaseAPIProvider):
    """OpenAI ChatGPT API provider"""
    
    def __init__(self, api_key: str, max_tokens: int = 4096, temperature: float = 0.7):
        super().__init__(api_key, max_tokens, temperature)
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-4o"  # Using GPT-4o for best performance
    
    @property
    def provider_name(self) -> str:
        return "OpenAI ChatGPT"
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             system_prompt: Optional[str] = None) -> APIResponse:
        """Generate chat completion using OpenAI API"""
        import aiohttp
        
        try:
            # Build messages with system prompt
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            formatted_messages.extend(messages)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        choice = data["choices"][0]
                        content = choice["message"]["content"]
                        
                        usage = data.get("usage", {})
                        
                        self._is_configured = True
                        
                        return APIResponse(
                            success=True,
                            content=content,
                            model=self.model,
                            usage={
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0)
                            }
                        )
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        logger.error(f"OpenAI API error: {response.status} - {error_msg}")
                        return APIResponse(
                            success=False,
                            content="",
                            error=f"API error: {response.status} - {error_msg}"
                        )
                        
        except Exception as e:
            logger.error(f"OpenAI API exception: {e}")
            return APIResponse(
                success=False,
                content="",
                error=str(e)
            )
    
    async def stream_completion(self, messages: List[Dict[str, str]],
                               system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream completion from OpenAI API"""
        import aiohttp
        
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            formatted_messages.extend(messages)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        yield f"Error: {error_msg}"
                        return
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data = line[6:]
                            
                            if data == '[DONE]':
                                return
                            
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def validate_connection(self) -> bool:
        """Validate OpenAI API connection"""
        import aiohttp
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            import asyncio
            response = asyncio.run(
                self._async_validate(headers)
            )
            
            if response.status == 200:
                self._is_configured = True
                return True
            return False
            
        except Exception as e:
            logger.error(f"OpenAI validation error: {e}")
            return False
    
    async def _async_validate(self, headers: dict):
        """Async helper for validation"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            return await session.get(
                "https://api.openai.com/v1/models",
                headers=headers
            )


class APIProviderFactory:
    """Factory for creating and managing API providers"""
    
    _providers: Dict[AIProvider, BaseAPIProvider] = {}
    
    @classmethod
    def create_provider(cls, provider: AIProvider, api_key: str, 
                       max_tokens: int = 4096, temperature: float = 0.7) -> BaseAPIProvider:
        """Create an API provider instance
        
        Args:
            provider: Type of AI provider
            api_key: API key for authentication
            max_tokens: Maximum tokens in response
            temperature: Response creativity
            
        Returns:
            Configured API provider instance
        """
        if provider == AIProvider.GEMINI:
            provider_instance = GeminiProvider(api_key, max_tokens, temperature)
        elif provider == AIProvider.OPENAI:
            provider_instance = OpenAIProvider(api_key, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        cls._providers[provider] = provider_instance
        return provider_instance
    
    @classmethod
    def get_provider(cls, provider: AIProvider) -> Optional[BaseAPIProvider]:
        """Get existing provider instance
        
        Args:
            provider: Type of AI provider
            
        Returns:
            Provider instance or None
        """
        return cls._providers.get(provider)
    
    @classmethod
    def validate_all_connections(cls) -> Dict[AIProvider, bool]:
        """Validate all configured providers
        
        Returns:
            Dictionary of provider to validation status
        """
        results = {}
        for provider, instance in cls._providers.items():
            results[provider] = instance.validate_connection()
        return results
    
    @classmethod
    def get_available_providers(cls) -> List[AIProvider]:
        """Get list of available/configured providers"""
        return [p for p, inst in cls._providers.items() if inst.is_configured]
