"""
LLM Client for iAgent and MemRec
Supports Azure OpenAI API, OpenAI-compatible APIs (TogetherAI, Anyscale, vLLM, etc.),
and local open-source models loaded directly via HuggingFace transformers
(provider_name="local_hf", e.g. Qwen/Qwen2.5-7B-Instruct - no API key needed).
"""
import os
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


class LLMClient:
    """LLM client supporting multiple providers"""
    
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        model: str = "gpt-4o-mini",
        provider_name: str = "azure_openai",  # "azure_openai", "openai", or "local_hf"
        save_conversations: bool = False,
        conversation_log_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize LLM client

        Args:
            api_endpoint: API endpoint (Azure OpenAI or OpenAI-compatible)
            api_key: API key
            api_version: API version (for Azure OpenAI)
            model: Model name (for local_hf: a HuggingFace repo id, e.g. Qwen/Qwen2.5-7B-Instruct)
            provider_name: "azure_openai", "openai" (OpenAI-compatible APIs incl. vLLM servers),
                           or "local_hf" (open-source model loaded in-process via transformers,
                           no API key/server required)
            save_conversations: Whether to save conversation history
            conversation_log_path: Path to save conversation logs (JSONL format)
            device: Device for local_hf models (default: "auto")
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model = model
        self.provider_name = provider_name

        if provider_name == "local_hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading local open-source model {model} via transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model,
                dtype=torch.float16,
                device_map=device or "auto"
            )
            self.client = None
        else:
            from openai import AzureOpenAI, OpenAI

            # Get from env if not provided
            if not self.api_endpoint:
                if provider_name == "azure_openai":
                    self.api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                else:
                    self.api_endpoint = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')

            if not self.api_key:
                if provider_name == "azure_openai":
                    self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
                else:
                    self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('TOGETHER_API_KEY') or os.getenv('ANYSCALE_API_KEY')

            if not self.api_endpoint or not self.api_key:
                raise ValueError(
                    f"API credentials not provided for {provider_name}. "
                    f"Please set endpoint and api_key, or use environment variables."
                )

            # Initialize client based on provider
            if provider_name == "azure_openai":
                self.client = AzureOpenAI(
                    azure_endpoint=self.api_endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
            else:  # OpenAI-compatible (TogetherAI, Anyscale, vLLM, etc.)
                self.client = OpenAI(
                    base_url=self.api_endpoint,
                    api_key=self.api_key
                )

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        
        # Conversation logging
        self.save_conversations = save_conversations
        self.conversation_log_path = conversation_log_path
        self.conversation_count = 0
        
        if self.save_conversations and self.conversation_log_path:
            log_path = Path(self.conversation_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"💬 Conversation logging enabled: {log_path}")
    
    def _log_conversation(
        self,
        messages: List[Dict[str, str]],
        response: str,
        metadata: Optional[Dict] = None
    ):
        """Log a conversation to file (JSONL format)"""
        if not self.save_conversations or not self.conversation_log_path:
            return
        
        self.conversation_count += 1
        
        log_entry = {
            'id': self.conversation_count,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'messages': messages,
            'response': response,
            'metadata': metadata or {}
        }
        
        try:
            with open(self.conversation_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log conversation: {e}")
    
    def _generate_local_hf(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4000,
        json_schema: Optional[Dict] = None
    ) -> str:
        """Generate a response using the in-process HuggingFace model (open-source, no API)."""
        import torch

        if json_schema:
            # Local models don't support structured output constraints, so ask
            # for JSON in the prompt instead (same pattern as agent_rec_qwen.py).
            required = json_schema.get('schema', {}).get('required', list(json_schema.get('schema', {}).get('properties', {}).keys()))
            messages = messages + [{
                'role': 'user',
                'content': f"Respond with ONLY a valid JSON object with exactly these keys: {required}. No markdown, no explanation outside the JSON."
            }]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.hf_model.device)

        gen_kwargs = {'max_new_tokens': max_tokens}
        if temperature is not None and temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
        else:
            gen_kwargs['do_sample'] = False

        with torch.no_grad():
            outputs = self.hf_model.generate(**inputs, **gen_kwargs)

        input_len = inputs['input_ids'].shape[-1]
        gen_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        self.total_input_tokens += input_len
        self.total_output_tokens += len(gen_ids)
        self.total_requests += 1

        self._log_conversation(messages=messages, response=response, metadata={'temperature': temperature, 'max_tokens': max_tokens})
        return response

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4000,
        json_schema: Optional[Dict] = None,
        max_retries: int = 5
    ) -> str:
        """
        Generate response from LLM with exponential backoff retry
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (None = use API default ~1.0)
            max_tokens: Maximum tokens to generate (increased to 4000 to avoid truncation)
            json_schema: Optional JSON schema for structured output
            max_retries: Maximum number of retries for rate limit errors
            
        Returns:
            Generated text
        """
        if self.provider_name == "local_hf":
            return self._generate_local_hf(messages, temperature=temperature, max_tokens=max_tokens, json_schema=json_schema)

        import time

        kwargs = {
            'model': self.model,
            'messages': messages
        }
        
        # Use max_completion_tokens for gpt-5-nano, max_tokens for others
        # gpt-5-nano requires max_completion_tokens instead of max_tokens
        if 'gpt-5-nano' in self.model.lower() or 'nano' in self.model.lower():
            kwargs['max_completion_tokens'] = max_tokens
            # gpt-5-nano does NOT support custom temperature (only default 1.0)
            # Do NOT add temperature parameter for this model
        else:
            kwargs['max_tokens'] = max_tokens
            # Only add temperature if specified (for non-nano models)
            if temperature is not None:
                kwargs['temperature'] = temperature
        
        # Add JSON schema if provided
        if json_schema:
            kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': json_schema
            }
        
        # Retry with exponential backoff for rate limit errors
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                response = completion.choices[0].message.content
                
                # Track token usage (compatible with different field names: prompt/input, completion/output)
                if hasattr(completion, 'usage'):
                    usage = completion.usage
                    if usage:
                        # Azure / OpenAI may use different field names
                        prompt_tokens = getattr(usage, 'prompt_tokens', None)
                        completion_tokens = getattr(usage, 'completion_tokens', None)
                        # Compatible with new field names input_tokens/output_tokens
                        input_tokens = getattr(usage, 'input_tokens', None)
                        output_tokens = getattr(usage, 'output_tokens', None)
                        
                        if prompt_tokens is None and input_tokens is not None:
                            prompt_tokens = input_tokens
                        if completion_tokens is None and output_tokens is not None:
                            completion_tokens = output_tokens
                        
                        self.total_input_tokens += prompt_tokens or 0
                        self.total_output_tokens += completion_tokens or 0
                        self.total_requests += 1
                
                # Log conversation if enabled
                self._log_conversation(
                    messages=messages,
                    response=response,
                    metadata={
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'has_json_schema': json_schema is not None,
                        'usage': {
                            'prompt_tokens': getattr(completion.usage, 'prompt_tokens', None) if hasattr(completion, 'usage') else None,
                            'completion_tokens': getattr(completion.usage, 'completion_tokens', None) if hasattr(completion, 'usage') else None,
                        } if hasattr(completion, 'usage') else None
                    }
                )
                
                return response
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "RateLimitReached" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        wait_time = 2 ** attempt
                        print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit error after {max_retries} attempts")
                        raise
                
                # For other errors, print and raise immediately
                print(f"Error calling LLM: {e}")
                raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed after {max_retries} attempts")
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Strip markdown code fences and surrounding prose from a model
        response so it can be json.loads()'d - a no-op for text that is
        already pure JSON (the constrained-output providers).
        """
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        properties: Dict[str, Dict],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4000,
        debug_logger = None
    ) -> Dict:
        """
        Generate JSON response from LLM
        
        Args:
            messages: List of message dicts
            properties: JSON schema properties
            temperature: Sampling temperature
            max_tokens: Maximum tokens (increased to 4000 to avoid truncation)
            debug_logger: Optional function to log debug info
            
        Returns:
            Parsed JSON dict
        """
        # Debug log: record complete LLM input
        if debug_logger:
            debug_logger("\n>>> LLM INPUT (Messages) <<<")
            for i, msg in enumerate(messages, 1):
                debug_logger(f"[Message {i}] Role: {msg.get('role', 'unknown')}")
                content = msg.get('content', '')
                # If content is too long, show preview
                if len(content) > 2000:
                    debug_logger(f"Content (first 2000 chars):\n{content[:2000]}\n... (truncated, total {len(content)} chars)")
                else:
                    debug_logger(f"Content:\n{content}")
                debug_logger("")  # Empty line separator
            
            debug_logger(">>> LLM Schema (Properties) <<<")
            debug_logger(json.dumps(properties, indent=2, ensure_ascii=False))
        
        # Build JSON schema
        json_schema = {
            'name': 'response',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': properties,
                'required': list(properties.keys()),
                'additionalProperties': False
            }
        }
        
        response_text = self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema
        )
        
        try:
            response_dict = json.loads(self._extract_json(response_text))

            # Debug log: record LLM output
            if debug_logger:
                debug_logger("\n>>> LLM OUTPUT (Parsed JSON) <<<")
                debug_logger(json.dumps(response_dict, indent=2, ensure_ascii=False))
                debug_logger("")  # Empty line separator
            
            return response_dict
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {response_text}")
            if debug_logger:
                debug_logger(f"\n❌ JSON Parse Error: {e}")
                debug_logger(f"Raw response:\n{response_text}")
            raise
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get cumulative token usage statistics"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_requests': self.total_requests,
            'avg_input_tokens': self.total_input_tokens / self.total_requests if self.total_requests > 0 else 0,
            'avg_output_tokens': self.total_output_tokens / self.total_requests if self.total_requests > 0 else 0,
        }
    
    def reset_token_stats(self):
        """Reset token usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
