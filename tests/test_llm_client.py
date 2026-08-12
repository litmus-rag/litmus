import pytest

from litmus.llm.client import extract_json, extract_json_array


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown_fence():
    text = '```json\n{"a": 1, "b": [1, 2]}\n```'
    assert extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_fence_without_language_tag():
    text = '```\n{"x": true}\n```'
    assert extract_json(text) == {"x": True}


def test_extract_json_with_leading_prose():
    text = 'Sure, here is the JSON you asked for:\n{"key": "value"}\nHope that helps!'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_array():
    text = 'Here you go: ["a", "b", "c"] thanks'
    assert extract_json(text) == ["a", "b", "c"]


def test_extract_json_nested_braces():
    text = '{"outer": {"inner": 1}, "list": [{"a": 1}, {"b": 2}]}'
    assert extract_json(text) == {"outer": {"inner": 1}, "list": [{"a": 1}, {"b": 2}]}


def test_extract_json_unparseable_raises():
    with pytest.raises(ValueError):
        extract_json("this is not json at all")


def test_extract_json_array_bare_array():
    assert extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_json_array_unwraps_single_key_object():
    # Azure/OpenAI JSON mode can't return a bare top-level array, so models
    # wrap it under a single key when the prompt asks for "a JSON array".
    assert extract_json_array('{"json": ["a", "b", "c"]}') == ["a", "b", "c"]
    assert extract_json_array('{"array": ["x"]}') == ["x"]
    assert extract_json_array('{"terms": []}') == []


def test_extract_json_array_rejects_multi_key_object():
    with pytest.raises(ValueError):
        extract_json_array('{"a": [1], "b": [2]}')


def test_extract_json_array_rejects_non_list_value():
    with pytest.raises(ValueError):
        extract_json_array('{"json": "not a list"}')
