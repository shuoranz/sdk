from types import SimpleNamespace as NS

from llmbenchmark import providers


def test_anthropic_message_keeps_only_text_blocks():
    response = NS(
        type="message",
        id="msg_1",
        model="claude-opus-5-5",
        stop_reason="end_turn",
        content=[NS(type="thinking", thinking=""), NS(type="text", text="Paris")],
        usage=NS(input_tokens=12, output_tokens=3),
    )
    out = providers.extract(response)
    assert out["provider"] == "anthropic"
    assert out["model"] == "claude-opus-5-5"
    assert out["output"] == "Paris"
    assert (out["input_tokens"], out["output_tokens"]) == (12, 3)
    assert out["metadata"] == {"stop_reason": "end_turn", "response_id": "msg_1"}


def test_openai_chat_completion_dict():
    response = {
        "object": "chat.completion",
        "id": "chatcmpl-1",
        "model": "gpt-5",
        "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    out = providers.extract(response)
    assert out["provider"] == "openai"
    assert out["output"] == "Paris"
    assert (out["input_tokens"], out["output_tokens"]) == (10, 2)


def test_openai_responses_api_dict():
    response = {
        "object": "response",
        "model": "gpt-5",
        "status": "completed",
        "output": [{"content": [{"type": "output_text", "text": "Par"}, {"type": "output_text", "text": "is"}]}],
        "usage": {"input_tokens": 8, "output_tokens": 2},
    }
    out = providers.extract(response)
    assert out["output"] == "Paris"
    assert out["input_tokens"] == 8


def test_unknown_object_gives_empty_dict():
    assert providers.extract(object()) == {}
    assert providers.extract(None) == {}
