from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class PlanningInterface:
    """Unified planning interface for VAD-based refinement.

    This minimal interface is designed to be extracted from VAD outputs
    without modifying the main VAD head logic.
    """

    scene_token: torch.Tensor
    reference_plan: torch.Tensor
    candidate_plans: Optional[torch.Tensor] = None
    plan_confidence: Optional[torch.Tensor] = None
    safety_features: Optional[Dict[str, torch.Tensor]] = None
    hard_case_score: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to(self, device: torch.device) -> "PlanningInterface":
        """Move all tensor fields to the target device."""
        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        return PlanningInterface(
            scene_token=_to(self.scene_token),
            reference_plan=_to(self.reference_plan),
            candidate_plans=_to(self.candidate_plans) if self.candidate_plans is not None else None,
            plan_confidence=_to(self.plan_confidence) if self.plan_confidence is not None else None,
            safety_features={k: _to(v) for k, v in self.safety_features.items()} if self.safety_features is not None else None,
            hard_case_score=_to(self.hard_case_score) if self.hard_case_score is not None else None,
            metadata=self.metadata.copy() if self.metadata is not None else {},
        )

    def __post_init__(self):
        assert isinstance(self.scene_token, torch.Tensor), 'scene_token must be a torch.Tensor'
        assert isinstance(self.reference_plan, torch.Tensor), 'reference_plan must be a torch.Tensor'

    def describe(self) -> str:
        """Return a short description of the interface shapes."""
        lines = [
            f'scene_token: {tuple(self.scene_token.shape)}',
            f'reference_plan: {tuple(self.reference_plan.shape)}',
        ]
        if self.candidate_plans is not None:
            lines.append(f'candidate_plans: {tuple(self.candidate_plans.shape)}')
        if self.plan_confidence is not None:
            lines.append(f'plan_confidence: {tuple(self.plan_confidence.shape)}')
        if self.safety_features is not None:
            lines.append(f'safety_features: {list(self.safety_features.keys())}')
        if self.hard_case_score is not None:
            lines.append(f'hard_case_score: {tuple(self.hard_case_score.shape)}')
        return '\n'.join(lines)

    @classmethod
    def collate(cls, interfaces: list["PlanningInterface"]) -> "PlanningInterface":
        """Collate a list of PlanningInterface into a batched one."""
        if not interfaces:
            raise ValueError("Cannot collate empty list of interfaces")

        # Stack scene_token
        scene_tokens = torch.stack([iface.scene_token for iface in interfaces])

        # Stack reference_plan
        reference_plans = torch.stack([iface.reference_plan for iface in interfaces])

        # Handle candidate_plans
        candidate_plans = None
        if all(iface.candidate_plans is not None for iface in interfaces):
            candidate_plans = torch.stack([iface.candidate_plans for iface in interfaces])

        # Handle plan_confidence
        plan_confidence = None
        if all(iface.plan_confidence is not None for iface in interfaces):
            plan_confidence = torch.stack([iface.plan_confidence for iface in interfaces])

        # Handle safety_features (dict of tensors)
        safety_features = None
        if all(iface.safety_features is not None for iface in interfaces):
            safety_keys = interfaces[0].safety_features.keys()
            safety_features = {}
            for key in safety_keys:
                if all(key in iface.safety_features for iface in interfaces):
                    safety_features[key] = torch.stack([iface.safety_features[key] for iface in interfaces])

        # Handle hard_case_score
        hard_case_score = None
        if all(iface.hard_case_score is not None for iface in interfaces):
            hard_case_score = torch.stack([iface.hard_case_score for iface in interfaces])

        # Merge metadata (take first one, or create empty)
        metadata = interfaces[0].metadata.copy() if interfaces[0].metadata else {}

        return cls(
            scene_token=scene_tokens,
            reference_plan=reference_plans,
            candidate_plans=candidate_plans,
            plan_confidence=plan_confidence,
            safety_features=safety_features,
            hard_case_score=hard_case_score,
            metadata=metadata,
        )
