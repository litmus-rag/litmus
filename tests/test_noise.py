from litmus.generate.noise import inject_typos


def test_inject_typos_changes_text():
    question = "What is the maximum number of users allowed on the Enterprise plan?"
    mutated = inject_typos(question, num_typos=1, seed=1)
    assert mutated != question


def test_inject_typos_is_deterministic_with_seed():
    question = "How do I reset my API key for the account?"
    a = inject_typos(question, num_typos=1, seed=42)
    b = inject_typos(question, num_typos=1, seed=42)
    assert a == b


def test_inject_typos_preserves_word_count():
    question = "What is the return policy for enterprise customers?"
    mutated = inject_typos(question, num_typos=1, seed=3)
    assert len(mutated.split(" ")) == len(question.split(" "))


def test_inject_typos_short_question_does_not_crash():
    assert inject_typos("Hi", num_typos=1, seed=0) == "Hi"


def test_inject_typos_empty_string():
    assert inject_typos("", num_typos=1, seed=0) == ""


def test_inject_typos_num_typos_zero_no_change():
    question = "What is the maximum file upload size on the free plan?"
    assert inject_typos(question, num_typos=0, seed=0) == question
