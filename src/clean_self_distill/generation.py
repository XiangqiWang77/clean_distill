"""Shared on-policy generation for Base, OPSD, and LGSD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .runtime import backbone_forward, input_device, project_logits, render_chat


EVALUATION_PROMPT_VERSION = "explicit-generation-budget-v1"


def problem_prompt(tokenizer, problem: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final "
                "answer within \\boxed{}."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def evaluation_problem_prompt(
    tokenizer, problem: str, *, max_new_tokens: int
) -> str:
    """Render the common answer-completion instruction used only at evaluation."""
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step. You have a strict "
                f"response budget of at most {int(max_new_tokens):,} generated tokens. "
                "Finish the reasoning and put the final answer within \\boxed{} before "
                "reaching that limit; prioritize completing the answer over extending "
                "the analysis."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = filtered
    return torch.multinomial(
        F.softmax(logits, dim=-1), num_samples=1, generator=generator
    ).squeeze(-1)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    problem: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    prompt_override: Optional[str] = None,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Generate one ordinary model response without a query-local adapter."""
    was_training = model.training
    model.eval()
    device = input_device(model)
    prompt = prompt_override or problem_prompt(tokenizer, problem)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ].to(device)
    generated: list[torch.Tensor] = []
    past_key_values = None
    next_input = prompt_ids
    generator: Optional[torch.Generator] = None

    for _ in range(max_new_tokens):
        hidden_sequence, past_key_values = backbone_forward(
            model,
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = project_logits(model, hidden_sequence[:, -1, :])
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed)
        token = _sample_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        generated.append(token)
        next_input = token[:, None].to(device)
        if (
            tokenizer.eos_token_id is not None
            and int(token.item()) == tokenizer.eos_token_id
        ):
            break

    response_ids = (
        torch.stack(generated, dim=1)
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    ).to(prompt_ids.device)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    model.train(was_training)
    return response, prompt_ids, response_ids
