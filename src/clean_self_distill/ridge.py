"""Closed-form query-local LM-head specialization.

For support hidden states H and a sparse desired logit residual R, we solve

    C = (H H^T + lambda I)^-1 R

and apply the equivalent update without materializing a dense vocabulary head:

    delta_logits(h) = (h H^T) C.

Every proposed and verified candidate is used. There is no Fit/Check split.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from .io import iter_rows, stable_hash, write_jsonl
from .runtime import backbone_forward, input_device, load_hf_model, project_logits, render_chat


def problem_prompt(tokenizer, problem: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def candidate_completion(candidate: dict[str, Any]) -> str:
    solution = str(candidate.get("solution", "")).strip()
    final_answer = str(candidate.get("final_answer", "")).strip()
    if not solution:
        raise ValueError("Every specialization candidate needs a solution")
    if final_answer and final_answer not in solution:
        solution = f"{solution}\n\nFinal answer: \\boxed{{{final_answer}}}"
    return solution


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _uniform_positions(length: int, max_positions: int) -> torch.Tensor:
    if length <= max_positions:
        return torch.arange(length, dtype=torch.long)
    # Uniform coverage keeps both reasoning transitions and the final answer.
    return torch.linspace(0, length - 1, steps=max_positions).round().long().unique()


@dataclass
class SparseRidgeAdapter:
    support_hidden: torch.Tensor
    coefficients: torch.Tensor
    vocab_ids: torch.Tensor
    hidden_scale: float
    ridge_lambda_effective: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.support_hidden.shape[0])

    @property
    def adapted_vocab_size(self) -> int:
        return int(self.vocab_ids.numel())

    def to(self, device: torch.device | str) -> "SparseRidgeAdapter":
        return SparseRidgeAdapter(
            support_hidden=self.support_hidden.to(device=device, dtype=torch.float32),
            coefficients=self.coefficients.to(device=device, dtype=torch.float32),
            vocab_ids=self.vocab_ids.to(device),
            hidden_scale=self.hidden_scale,
            ridge_lambda_effective=self.ridge_lambda_effective,
            metadata=dict(self.metadata),
        )

    def selected_delta(self, hidden: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
        original_shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float() / self.hidden_scale
        support = self.support_hidden.to(flat_hidden.device, dtype=torch.float32)
        coefficients = self.coefficients.to(flat_hidden.device, dtype=torch.float32)
        chunks = []
        for start in range(0, flat_hidden.shape[0], chunk_size):
            query = flat_hidden[start : start + chunk_size]
            chunks.append((query @ support.T) @ coefficients)
        delta = torch.cat(chunks, dim=0) if chunks else flat_hidden.new_zeros((0, coefficients.shape[1]))
        return delta.reshape(*original_shape, coefficients.shape[1])

    def apply_to_logits(self, logits: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(f"Logit/hidden prefix shapes differ: {logits.shape} vs {hidden.shape}")
        adapted = logits.clone()
        delta = self.selected_delta(hidden).to(adapted.dtype)
        vocab_ids = self.vocab_ids.to(adapted.device)
        adapted[..., vocab_ids] = adapted[..., vocab_ids] + delta
        return adapted

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "clean-self-distill-ridge-v1",
            "support_hidden": self.support_hidden.detach().cpu(),
            "coefficients": self.coefficients.detach().cpu(),
            "vocab_ids": self.vocab_ids.detach().cpu(),
            "hidden_scale": self.hidden_scale,
            "ridge_lambda_effective": self.ridge_lambda_effective,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SparseRidgeAdapter":
        state = torch.load(path, map_location=map_location)
        if state.get("schema_version") != "clean-self-distill-ridge-v1":
            raise ValueError(f"Unsupported adapter schema in {path}")
        return cls(
            support_hidden=state["support_hidden"],
            coefficients=state["coefficients"],
            vocab_ids=state["vocab_ids"],
            hidden_scale=float(state["hidden_scale"]),
            ridge_lambda_effective=float(state["ridge_lambda_effective"]),
            metadata=dict(state.get("metadata", {})),
        )


@torch.inference_mode()
def _candidate_features(
    model,
    tokenizer,
    candidate: dict[str, Any],
    *,
    max_tokens: int,
    hard_negatives: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(
        problem_prompt(tokenizer, str(candidate["problem"])),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    completion_ids = tokenizer(
        candidate_completion(candidate),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if tokenizer.eos_token_id is not None:
        completion_ids = torch.cat([completion_ids, completion_ids.new_tensor([tokenizer.eos_token_id])])
    available = max_length - int(prompt_ids.numel())
    if available <= 0:
        raise ValueError("Candidate prompt exceeds max_length")
    completion_ids = completion_ids[:available]
    if completion_ids.numel() == 0:
        raise ValueError("Candidate completion tokenized to zero tokens")

    full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(input_device(model))
    all_hidden, _ = backbone_forward(
        model,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
    )
    prompt_len = int(prompt_ids.numel())
    completion_len = int(completion_ids.numel())
    start = prompt_len - 1
    hidden = all_hidden[0, start : start + completion_len]
    labels = completion_ids.to(hidden.device)

    positions = _uniform_positions(completion_len, max_tokens).to(hidden.device)
    hidden = hidden.index_select(0, positions)
    selected_logits = project_logits(model, hidden).float()
    hidden = hidden.float().cpu()
    labels = labels.index_select(0, positions)

    k = min(max(1, hard_negatives), selected_logits.shape[-1])
    top_values, top_ids = torch.topk(selected_logits, k=k, dim=-1)
    log_normalizer = torch.logsumexp(selected_logits, dim=-1, keepdim=True)
    top_probs = torch.exp(top_values - log_normalizer)
    target_logits = selected_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_probs = torch.exp(target_logits - log_normalizer.squeeze(-1))
    target_log_probs = target_logits - log_normalizer.squeeze(-1)
    return {
        "hidden": hidden,
        "labels": labels.cpu(),
        "top_ids": top_ids.cpu(),
        "top_probs": top_probs.cpu(),
        "target_probs": target_probs.cpu(),
        "target_log_probs": target_log_probs.cpu(),
    }


def _build_sparse_residual(
    labels: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
    target_probs: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_ids = torch.unique(torch.cat([labels.reshape(-1), top_ids.reshape(-1)])).sort().values
    id_to_column = {int(token_id): column for column, token_id in enumerate(vocab_ids.tolist())}
    residual = torch.zeros(labels.numel(), vocab_ids.numel(), dtype=torch.float32)
    for row in range(labels.numel()):
        for token_id, probability in zip(top_ids[row].tolist(), top_probs[row].tolist()):
            if int(token_id) == int(labels[row]):
                continue
            residual[row, id_to_column[int(token_id)]] -= step_size * float(probability)
        residual[row, id_to_column[int(labels[row])]] += step_size * (1.0 - float(target_probs[row]))
    return residual, vocab_ids


def _cholesky_solve(kernel: torch.Tensor, residual: torch.Tensor, ridge: float) -> torch.Tensor:
    regularized = kernel.clone()
    regularized.diagonal().add_(ridge)
    try:
        factor = torch.linalg.cholesky(regularized)
        return torch.cholesky_solve(residual, factor)
    except RuntimeError:
        return torch.linalg.solve(regularized, residual)


@torch.inference_mode()
def fit_ridge_adapter(
    model,
    tokenizer,
    candidates: list[dict[str, Any]],
    *,
    ridge_lambda: float = 0.1,
    residual_step_size: float = 0.8,
    max_tokens_per_candidate: int = 64,
    max_support_tokens: int = 256,
    hard_negatives: int = 8,
    max_length: int = 4096,
    query_id: str = "",
) -> tuple[SparseRidgeAdapter, dict[str, float]]:
    """Fit the temporary teacher using all verified proposed candidates."""
    if not candidates:
        raise ValueError("At least one specialization candidate is required")
    if max_support_tokens < len(candidates):
        raise ValueError(
            f"max_support_tokens={max_support_tokens} is smaller than "
            f"num_candidates={len(candidates)}"
        )
    device = input_device(model)
    _synchronize(device)
    total_start = time.perf_counter()
    feature_start = total_start
    base_allocation, remainder = divmod(max_support_tokens, len(candidates))
    token_allocations = [
        min(max_tokens_per_candidate, base_allocation + int(index < remainder))
        for index in range(len(candidates))
    ]
    features = []
    for candidate, token_budget in zip(candidates, token_allocations):
        features.append(
            _candidate_features(
                model,
                tokenizer,
                candidate,
                max_tokens=token_budget,
                hard_negatives=hard_negatives,
                max_length=max_length,
            )
        )
    _synchronize(device)
    feature_seconds = time.perf_counter() - feature_start

    hidden = torch.cat([item["hidden"] for item in features], dim=0).float()
    labels = torch.cat([item["labels"] for item in features], dim=0).long()
    top_ids = torch.cat([item["top_ids"] for item in features], dim=0).long()
    top_probs = torch.cat([item["top_probs"] for item in features], dim=0).float()
    target_probs = torch.cat([item["target_probs"] for item in features], dim=0).float()
    target_log_probs = torch.cat([item["target_log_probs"] for item in features], dim=0).float()
    residual, vocab_ids = _build_sparse_residual(
        labels,
        top_ids,
        top_probs,
        target_probs,
        residual_step_size,
    )

    # Scaling keeps the kernel O(1) across hidden widths. ridge_lambda is
    # relative to mean kernel diagonal, making it transferable across models.
    hidden_scale = math.sqrt(hidden.shape[-1])
    support_hidden = hidden / hidden_scale
    solve_start = time.perf_counter()
    kernel = support_hidden @ support_hidden.T
    ridge_effective = float(ridge_lambda * kernel.diagonal().mean().clamp(min=1e-8).item())
    coefficients = _cholesky_solve(kernel, residual, ridge_effective)
    solve_seconds = time.perf_counter() - solve_start
    # Deployment specialization ends here. Diagnostics below are deliberately
    # excluded from the reported adaptation wall time.
    total_seconds = time.perf_counter() - total_start

    adapter = SparseRidgeAdapter(
        support_hidden=support_hidden.to(torch.float16),
        coefficients=coefficients.to(torch.float16),
        vocab_ids=vocab_ids,
        hidden_scale=hidden_scale,
        ridge_lambda_effective=ridge_effective,
        metadata={
            "query_id": query_id,
            "num_candidates": len(candidates),
            "support_tokens": int(hidden.shape[0]),
            "max_support_tokens": max_support_tokens,
            "hard_negatives": hard_negatives,
            "residual_step_size": residual_step_size,
            "uses_all_candidates": True,
            "fit_check_split": False,
            "teacher_context_sources": ["proposed_candidate_problem", "proposed_candidate_solution"],
        },
    )

    # Measure how much the closed-form update fits its own proposed set.
    support_device = device
    h = hidden.to(support_device)
    base_target_lp = target_log_probs.to(support_device)
    delta = adapter.to(support_device).selected_delta(h)
    columns = {int(token_id): index for index, token_id in enumerate(vocab_ids.tolist())}
    label_columns = torch.tensor([columns[int(label)] for label in labels], device=support_device)
    # The exact adapted normalizer requires full logits. This selected-vocab
    # proxy is reported only as a fit diagnostic, never as target evaluation.
    target_delta = delta.gather(-1, label_columns.unsqueeze(-1)).squeeze(-1)
    support_margin_gain = target_delta.mean().item()
    _synchronize(device)
    metrics = {
        "specialization_seconds": total_seconds,
        "feature_extraction_seconds": feature_seconds,
        "closed_form_solve_seconds": solve_seconds,
        "support_tokens": float(hidden.shape[0]),
        "adapted_vocab_size": float(vocab_ids.numel()),
        "adapter_rank": float(adapter.rank),
        "ridge_lambda_effective": ridge_effective,
        "proposal_fit_target_logit_gain": support_margin_gain,
        "proposal_base_target_nll": float((-base_target_lp).mean().item()),
    }
    return adapter, metrics


def _safe_filename(query_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id).strip("_")
    return (slug[:80] or "query") + "-" + stable_hash(query_id) + ".pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--residual-step-size", type=float, default=0.8)
    parser.add_argument("--max-tokens-per-candidate", type=int, default=64)
    parser.add_argument("--max-support-tokens", type=int, default=256)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-specialization-candidates", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--allow-hindsight-exposure",
        action="store_true",
        help="Permit a proposal manifest marked as using target answer/solution (ablation only)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rows = list(iter_rows(args.proposals))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    for index, row in enumerate(tqdm(rows, desc="closed-form specialization")):
        query_id = str(row["query_id"])
        firewall = row.get("firewall_audit", {})
        if isinstance(firewall, dict):
            exposed = any(
                str(firewall.get(key, False)).strip().lower() in {"1", "true", "yes"}
                for key in ("target_answer_loaded", "target_solution_loaded")
            )
            if exposed and not args.allow_hindsight_exposure:
                raise ValueError(
                    f"{query_id} is marked as hindsight-contaminated; use "
                    "--allow-hindsight-exposure only for an ablation"
                )
        candidates = list(row.get("specialization_candidates", []))
        if args.num_specialization_candidates is not None:
            candidates = candidates[: args.num_specialization_candidates]
        adapter, metrics = fit_ridge_adapter(
            model,
            tokenizer,
            candidates,
            ridge_lambda=args.ridge_lambda,
            residual_step_size=args.residual_step_size,
            max_tokens_per_candidate=args.max_tokens_per_candidate,
            max_support_tokens=args.max_support_tokens,
            hard_negatives=args.hard_negatives,
            max_length=args.max_length,
            query_id=query_id,
        )
        filename = _safe_filename(query_id)
        adapter.save(output_dir / filename)
        manifest = {
            "query_id": query_id,
            "adapter_path": filename,
            "problem_sha256": row.get("problem_sha256", stable_hash(str(row.get("problem", "")), 64)),
            "model": args.model,
            **metrics,
        }
        write_jsonl(manifest_path, [manifest], append=index > 0)


if __name__ == "__main__":
    main()
