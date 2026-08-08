from types import SimpleNamespace

from src.clean_self_distill.generation import (
    EVALUATION_PROMPT_VERSION,
    evaluation_problem_prompt,
)


def test_evaluation_prompt_records_and_states_the_generation_budget() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    prompt = evaluation_problem_prompt(
        tokenizer,
        "Compute 1+1.",
        max_new_tokens=10_240,
    )

    assert EVALUATION_PROMPT_VERSION == "explicit-generation-budget-v1"
    assert "at most 10,240 generated tokens" in prompt
    assert "Finish the reasoning" in prompt
    assert "\\boxed{}" in prompt
    assert prompt.endswith("ASSISTANT:")
