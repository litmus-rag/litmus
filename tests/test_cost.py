from litmus.llm.cost import count_tokens, estimate_llm_cost_usd


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_positive_for_nonempty():
    assert count_tokens("hello world") > 0


def test_count_tokens_scales_with_length():
    short = count_tokens("hello")
    long = count_tokens("hello " * 50)
    assert long > short


def test_estimate_llm_cost_usd_nonnegative():
    cost = estimate_llm_cost_usd("azure/gpt-5.4", 1000, 500)
    assert cost >= 0


def test_estimate_llm_cost_usd_scales_with_tokens():
    small = estimate_llm_cost_usd("gpt-4o", 100, 50)
    large = estimate_llm_cost_usd("gpt-4o", 10000, 5000)
    assert large > small


def test_estimate_llm_cost_usd_unknown_model_uses_fallback():
    cost = estimate_llm_cost_usd("totally-unknown-model-xyz", 1000, 1000)
    assert cost > 0
