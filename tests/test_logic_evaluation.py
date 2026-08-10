import json
import sys
from pathlib import Path

from src.clean_self_distill.logic_evaluation import (
    canonicalize_formula,
    extract_binary_answer,
    extract_countermodel,
    extract_formula,
    extract_option_set,
    serialize_countermodel,
    score_logicskills,
)

def test_answer_extractors_prefer_explicit_final_answer():
    response = "<think>trial 000 and option 2</think>\nFinal answer: \\boxed{10101}"
    assert extract_binary_answer(response, 5) == "10101"
    assert extract_option_set("Reasoning\nFinal answer: [2, 6]") == [2, 6]
    assert extract_formula("```text\n∀x(Fx→Gx)\n```") == "∀x(Fx→Gx)"


def test_answer_extractors_handle_gptoss_harmony_final_channel():
    response = (
        "analysis provisional option 2 and formula Fa"
        "assistantfinal**Answer: Only statement 6 must be true.**"
    )
    assert extract_option_set(response) == [6]
    assert extract_binary_answer("analysis 00000assistantfinal10101", 5) == "10101"


def test_formula_canonicalizer_accepts_standard_model_notation():
    assert canonicalize_formula("P(c,a) ∧ ¬∀x (M(x) → Q(x,a))") == (
        "(Pca∧¬∀x(Mx→Qxa))"
    )
    assert canonicalize_formula("(∀x(Nx→Jx))→((¬Qac)→∀x(Nx→¬Lx))") == (
        "(∀x(Nx→Jx)→(¬Qac→∀x(Nx→¬Lx)))"
    )


def test_countermodel_parser_supports_documented_line_format():
    response = """Domain: [0, 1, 2]
Constants:
"a": 0
Monadic predicates:
"F": [0, 2]
Binary predicates:
"R": [[0, 1], [2, 0]]
"""
    assert extract_countermodel(response) == {
        "Domain": [0, 1, 2],
        "a": 0,
        "F": [0, 2],
        "R": [[0, 1], [2, 0]],
    }


def test_countermodel_parser_handles_gptoss_markdown_final_channel():
    response = """analysis draft
assistantfinal**Domain**: [0, 1, 2]

**Constants**
- `a`: 0

**Monadic predicates**
- `F`: [0, 2]

**Binary predicates**
- `R`: [[0, 1], [2, 0]]
"""
    assert extract_countermodel(response) == {
        "Domain": [0, 1, 2],
        "a": 0,
        "F": [0, 2],
        "R": [[0, 1], [2, 0]],
    }


def test_countermodel_serialization_handles_python_set_literals():
    assert serialize_countermodel({"Domain": {2, 0, 1}, "F": {(2, 1)}}) == (
        '{"Domain": [0, 1, 2], "F": [[2, 1]]}'
    )


def test_logicskills_symbolization_uses_z3_verifier():
    scratch = Path("/home/da839/scratch_pi_mg269/da839/clean_distill")
    deps = scratch / "data/sources/python-deps"
    repo = scratch / "data/sources/LogicSkills"
    sys.path[:0] = [str(deps), str(repo)]
    expected = "(Pca∧∃x(Mx∧¬Qxa))"
    correct, extracted = score_logicskills(
        expected,
        task="symbolization",
        target=expected,
        payload={"expected_formula": expected, "correct_options": []},
        logic_repo=repo,
    )
    assert correct is True
    assert extracted == expected


def test_logic_dataset_manifest_is_complete_and_disjoint():
    path = Path(
        "/home/da839/scratch_pi_mg269/da839/clean_distill/data/verl/LOGIC_DATA_MANIFEST.json"
    )
    manifest = json.loads(path.read_text())
    assert manifest["outputs"]["satquest_eval"]["rows"] == 3360
    assert manifest["outputs"]["logicskills_eval"]["rows"] == 1500
    assert manifest["split_contract"]["satquest_train_eval_id_overlap"] == 0
