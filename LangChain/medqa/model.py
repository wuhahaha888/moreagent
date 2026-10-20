"""明确发送兼容接口参数，避免 SDK 框架重命名输出 token 上限。"""
from types import SimpleNamespace
from urllib.parse import urlparse


class CompatibleChat:
    def __init__(self, *, model, base_url, api_key, max_tokens, timeout_seconds=60, client=None):
        if client is None:
            from openai import OpenAI
            options={}
            if urlparse(base_url).hostname in {'127.0.0.1','localhost','::1'}:
                import httpx
                # Cluster HTTP_PROXY settings must never intercept localhost.
                options['http_client']=httpx.Client(trust_env=False,timeout=timeout_seconds)
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=0,**options)
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.timeout_seconds = timeout_seconds
        # 仅官方 DeepSeek 主机使用供应商扩展；校内服务不被强加此字段。
        self.deepseek = urlparse(base_url).hostname == "api.deepseek.com"
        self.qwen3_nonthinking = (urlparse(base_url).hostname in {'127.0.0.1','localhost','::1'}
                                  and model.lower().rsplit('/', 1)[-1].startswith('qwen3-'))

    def invoke(self, messages):
        return self.invoke_bounded(messages)

    def invoke_bounded(self, messages, *, remaining_seconds=None, remaining_tokens=None):
        output_limit = min(self.max_tokens, max(1, int(remaining_tokens))) if remaining_tokens is not None else self.max_tokens
        params = {"model": self.model, "max_tokens": output_limit, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system" if m.type == "system" else "user", "content": m.content}
                               for m in messages]}
        if self.deepseek:
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        elif self.qwen3_nonthinking:
            params['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        if remaining_seconds is not None:
            params["timeout"] = max(.01, min(self.timeout_seconds, remaining_seconds))
        response = self.client.chat.completions.create(**params)
        usage = response.usage.model_dump() if response.usage else {}
        return SimpleNamespace(content=response.choices[0].message.content or "",
                               response_metadata={"token_usage": usage,
                                                  "model": getattr(response, "model", "unknown"),
                                                  "finish_reason": response.choices[0].finish_reason})
