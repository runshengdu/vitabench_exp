import json
import re
import time
from typing import Any, Optional

from loguru import logger
import requests


from vita.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add interleaved thinking content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                # Prefer reasoning_details (OpenAI/Anthropic/Google) over reasoning_content (DeepSeek)
                reasoning_details = message.raw_data["message"].get("reasoning_details")
                if reasoning_details is not None:
                    messages_formatted[-1]["reasoning_details"] = reasoning_details
                else:
                    messages_formatted[-1]["reasoning_content"] = message.raw_data["message"].get("reasoning_content")
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        # Find the last AssistantMessage (skip trailing ToolMessages)
        idx = -1
        while abs(idx) <= len(messages):
            if isinstance(messages[idx], AssistantMessage):
                break
            idx -= 1
        else:
            # No AssistantMessage found, return as-is
            return messages_formatted
        
        content = [
            {
                "type": "text",
                "text": messages[idx].content
            }
        ]
        if messages[idx].raw_data and messages[idx].raw_data.get("message", {}).get("tool_calls", []):
            content.append(
                {
                    "type": "tool_use",
                    "id": messages[idx].raw_data["message"]["tool_calls"][0]['id'],
                    "name": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["name"],
                    "input": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["arguments"]
                }
            )
        reasoning_content = None
        if messages[idx].raw_data and messages[idx].raw_data.get("message"):
            reasoning_content = messages[idx].raw_data["message"].get("reasoning_content", None) or messages[idx].raw_data["message"].get("reasoning", None)
        if reasoning_content:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning_content
                }
            )

        messages_formatted[idx]["content"] = content
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    return messages_formatted


def apply_anthropic_prompt_cache(
    messages_formatted: list[dict],
    raw_messages: list[Message],
    turn_interval: int = 3,
    max_breakpoints: int = 4,
) -> list[dict]:
    """Apply Anthropic-style prompt caching for Claude via OpenRouter.

    This converts system/user/assistant messages into multipart content with a
    cache_control marker on the text block so that Anthropic/OpenRouter can
    reuse the expensive context across requests.
    """
    breakpoints_used = 0

    for raw_msg, formatted_msg in zip(raw_messages, messages_formatted):
        if breakpoints_used >= max_breakpoints:
            break

        role = getattr(raw_msg, "role", None)
        turn_idx = getattr(raw_msg, "turn_idx", None)

        if role not in ("system", "user", "assistant"):
            continue

        # Require a valid turn index and only cache every `turn_interval` turns.
        if turn_idx is None:
            # Fallback: allow caching for system messages without explicit turn index.
            if role == "system":
                turn_idx = 0
            else:
                continue

        if turn_idx % turn_interval != 0:
            continue

        content = formatted_msg.get("content")
        if not content:
            continue

        # If the content is a plain string, wrap it as a single text block
        # with an ephemeral cache_control marker.
        if isinstance(content, str):
            formatted_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            breakpoints_used += 1
        # If the content is already a multipart list, attach cache_control to
        # the first suitable text block that doesn't already specify it.
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "text":
                    continue
                if "cache_control" in part:
                    continue
                part["cache_control"] = {"type": "ephemeral"}
                breakpoints_used += 1
                break

    return messages_formatted


def kwargs_adapter(data: dict, enable_think: bool, messages: list) -> dict:
    model_name = str(data.get("model", "")).lower()
    if "claude" in model_name:
        # Claude via OpenRouter: enable reasoning tokens using unified `reasoning` parameter
        if enable_think:
            # Ensure we have a reasonable max_tokens for both output and reasoning budget
            max_tokens = data.get("max_tokens")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                # Use a conservative default if not provided by caller/config
                max_tokens = 8192
                data["max_tokens"] = max_tokens

            # If caller hasn't already set a reasoning config, enable medium-effort reasoning
            if "reasoning" not in data:
                data["reasoning"] = {"effort": "medium"}

            # Additionally, for Claude we still transform the last assistant message
            # into Anthropic's official content format (with optional thinking block)
            data["messages"] = to_claude_think_official(data["messages"], messages)
        else:
            # Thinking disabled: remove any explicit reasoning config if present
            data.pop("reasoning", None)

        # For Anthropic Claude via OpenRouter, enable prompt caching on system
        # messages by inserting cache_control markers into the text content.
        data["messages"] = apply_anthropic_prompt_cache(data["messages"], messages)
    elif "minimax" in model_name:
        # MiniMax models: support Anthropic-compatible prompt caching
        # See: https://platform.minimaxi.com/docs/api-reference/text-prompt-caching
        # MiniMax uses the same cache_control: {"type": "ephemeral"} format as Anthropic
        data["messages"] = apply_anthropic_prompt_cache(data["messages"], messages)
        # Non-Claude models: use a generic top-level thinking flag without changing messages
        if enable_think:
            data["thinking"] = {"type": "enabled"}
        else:
            data["thinking"] = {"type": "disabled"}
    else:
        # Non-Claude models: use a generic top-level thinking flag without changing messages
        if enable_think:
            data["thinking"] = {"type": "enabled"}
        else:
            data["thinking"] = {"type": "disabled"}
    return data


def _parse_stream_response(response: requests.Response, model: str) -> dict:
    """
    Parse streaming response and aggregate chunks into a complete response.
    
    Args:
        response: The streaming response object.
        model: The model name for cost calculation.
    
    Returns:
        Aggregated response dictionary in the same format as non-streaming response.
    """
    content_parts = []
    reasoning_parts = []  # For string-based reasoning
    reasoning_acc = {}  # For structured reasoning: (index, type, format) -> accumulated dict
    reasoning_order = []  # Track order of reasoning entries
    tool_calls_dict = {}  # id -> {id, function: {name, arguments}}
    usage = None
    role = "assistant"
    
    for line in response.iter_lines():
        if not line:
            continue
        
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        
        # Skip non-data lines
        if not line_str.startswith('data: '):
            continue
        
        data_str = line_str[6:]  # Remove 'data: ' prefix
        
        # Skip [DONE] marker
        if data_str.strip() == '[DONE]':
            continue
        
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        
        # Extract usage if present (usually in the last chunk)
        if 'usage' in chunk and chunk['usage']:
            usage = chunk['usage']
        
        # Process choices
        choices = chunk.get('choices', [])
        if not choices:
            continue
        
        delta = choices[0].get('delta', {})
        
        # Get role if present
        if 'role' in delta:
            role = delta['role']
        
        # Aggregate content
        if 'content' in delta and delta['content']:
            content_parts.append(delta['content'])
        
        # Aggregate reasoning content (for thinking models)
        # reasoning_content: string (DeepSeek)
        # reasoning_details: list of dicts (OpenAI/Anthropic/Google)
        delta_reasoning_content = delta.get('reasoning_content')
        if delta_reasoning_content and isinstance(delta_reasoning_content, str):
            reasoning_parts.append(delta_reasoning_content)
        
        delta_reasoning_details = delta.get('reasoning_details')
        if delta_reasoning_details and isinstance(delta_reasoning_details, list):
            # List-based structured reasoning (e.g., OpenAI/Gemini/Anthropic)
            # Aggregate entries by (index, type, format) and concatenate their string fields
            for item in delta_reasoning_details:
                if not isinstance(item, dict):
                    continue
                
                key = (
                    item.get("index"),
                    item.get("type"),
                    item.get("format"),
                )
                
                if key not in reasoning_acc:
                    reasoning_acc[key] = {}
                    reasoning_order.append(key)
                
                acc_item = reasoning_acc[key]
                
                # Copy all fields, concatenating string content fields
                for k, v in item.items():
                    if k in ("summary", "text", "data", "signature"):
                        # Concatenate content string fields across chunks
                        if isinstance(v, str):
                            prev = acc_item.get(k, "") or ""
                            acc_item[k] = prev + v
                    elif k not in acc_item:
                        # Copy other fields once
                        acc_item[k] = v
        
        # Aggregate tool calls
        if 'tool_calls' in delta and delta['tool_calls']:
            for tc in delta['tool_calls']:
                tc_index = tc.get('index', 0)
                if tc_index not in tool_calls_dict:
                    tool_calls_dict[tc_index] = {
                        'id': tc.get('id', ''),
                        'type': 'function',
                        'function': {
                            'name': '',
                            'arguments': ''
                        }
                    }
                
                if tc.get('id'):
                    tool_calls_dict[tc_index]['id'] = tc['id']
                
                if 'function' in tc:
                    if tc['function'].get('name'):
                        tool_calls_dict[tc_index]['function']['name'] = tc['function']['name']
                    if tc['function'].get('arguments'):
                        tool_calls_dict[tc_index]['function']['arguments'] += tc['function']['arguments']
    
    # Build aggregated response
    aggregated_content = ''.join(content_parts) if content_parts else None
    
    # Build aggregated reasoning
    # reasoning_content: string 
    aggregated_reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
    # reasoning_details: list of dicts (OpenAI/Anthropic/Google)
    aggregated_reasoning_details = [reasoning_acc[k] for k in reasoning_order] if reasoning_order else None
    
    # Convert tool_calls_dict to list
    tool_calls_list = None
    if tool_calls_dict:
        tool_calls_list = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
    
    # Build message
    message = {
        'role': role,
        'content': aggregated_content,
        'tool_calls': tool_calls_list
    }
    
    if aggregated_reasoning_content:
        message['reasoning_content'] = aggregated_reasoning_content
    if aggregated_reasoning_details:
        message['reasoning_details'] = aggregated_reasoning_details
    
    # Build final response structure
    aggregated_response = {
        'choices': [{
            'message': message,
            'finish_reason': 'stop'
        }],
        'usage': usage or {'prompt_tokens': 0, 'completion_tokens': 0}
    }
    
    return aggregated_response


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model using streaming to avoid gateway timeout.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        try:
            data = {
                "model": model,
                "messages": messages_formatted,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": kwargs.get("temperature"),
                "tools": tools,
                "tool_choice": tool_choice,
            }
            data.update(models[model])
            data = kwargs_adapter(data, enable_think, messages)
            headers = models[model]["headers"]

            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(
                        data["base_url"], 
                        json=data, 
                        headers=headers, 
                        timeout=(10, 600),
                        stream=True
                    )

                    if response.status_code != 500:
                        # Parse streaming response and aggregate
                        response = _parse_stream_response(response, model)
                        break

                    if attempt < max_retries:
                        logger.warning(f"API returned 500 error, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        response.raise_for_status()

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        except Exception as e:
            logger.error(e)
            raise e
        usage = get_response_usage(response)
        cost = get_response_cost(usage, model)
        try:
            response = response['choices'][0]
        except:
            print(f"bad response: {response}")
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        content = response['message'].get('content')
        # Prefer reasoning_details (OpenAI/Anthropic/Google) over reasoning_content (DeepSeek)
        reasoning_content = (
            response['message'].get('reasoning_details')
            or response['message'].get('reasoning_content')
            or response['message'].get('reasoning')
        )
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=response,
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)
        raise RuntimeError(f"LLM API call failed after all retries: {e}") from e


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost
