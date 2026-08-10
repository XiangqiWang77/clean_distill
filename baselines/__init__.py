"""Paper-faithful comparison baselines for the clean-distillation study."""

from .objectives import (
    DemoPSDMetrics,
    SRPOMetrics,
    demopsd_reverse_kl,
    grpo_group_advantages,
    grpo_token_loss,
    srpo_entropy_weights,
    srpo_route_masks,
    srpo_topk_jsd,
)

__all__ = [
    "DemoPSDMetrics",
    "SRPOMetrics",
    "demopsd_reverse_kl",
    "grpo_group_advantages",
    "grpo_token_loss",
    "srpo_entropy_weights",
    "srpo_route_masks",
    "srpo_topk_jsd",
]
