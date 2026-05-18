"""Identity normalization tests."""

from backend.services.identity import (
    heuristic_canonical_id, normalize_family, parse_parameter_size, parse_variant,
    resolve_alias, all_canonical_models,
)


def test_parse_parameter_size():
    assert parse_parameter_size("Llama-3.1-8B-Instruct") == "8B"
    assert parse_parameter_size("Mixtral-8x7B-Instruct-v0.1") == "8x7B"
    assert parse_parameter_size("Phi-3.5-mini-3.8B") == "3.8B"
    assert parse_parameter_size("CodeLlama-13b") == "13B"
    assert parse_parameter_size("no-numbers") == ""


def test_parse_variant():
    assert parse_variant("Llama-3.1-8B-Instruct") == "instruct"
    assert parse_variant("gemma-2-9b-it") == "instruct"
    assert parse_variant("llava-vision") == "vision"
    assert parse_variant("Llama-3.1-8B") == "base"


def test_heuristic_canonical_id():
    cid = heuristic_canonical_id("Meta-Llama-3.1-8B-Instruct")
    assert cid is not None
    assert "8b" in cid.lower()
    assert "instruct" in cid

    cid = heuristic_canonical_id("Mixtral-8x7B-Instruct-v0.1")
    assert cid is not None
    assert "8x7b" in cid.lower()


def test_curated_aliases_loaded():
    models = all_canonical_models()
    assert len(models) >= 20
    families = {m.family for m in models}
    assert "llama-3.1" in families
    assert "qwen-2.5" in families


def test_resolve_alias_known():
    assert resolve_alias("ollama", "llama3.1:8b") == "llama-3.1:8b:instruct"
    assert resolve_alias("hf", "Qwen/Qwen2.5-7B-Instruct") == "qwen-2.5:7b:instruct"


def test_resolve_alias_unknown_returns_none():
    assert resolve_alias("ollama", "totally-fake-model:99b") is None
