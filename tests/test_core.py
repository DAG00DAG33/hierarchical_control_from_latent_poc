from __future__ import annotations

import numpy as np
import torch

from hcl_poc.config import Config
from hcl_poc.eval import horizon_steps
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel, ObservationEncoder, RepresentationWorldModel
from hcl_poc.utils import Standardizer


def test_horizon_seconds_to_steps() -> None:
    config = Config(raw={"control_freq": 20}, path="dummy.yaml")  # type: ignore[arg-type]
    assert horizon_steps(config, 0.25) == 5
    assert horizon_steps(config, 1.0) == 20
    assert horizon_steps(config, 4.0) == 80


def test_standardizer_round_trip() -> None:
    x = np.random.randn(32, 5).astype(np.float32)
    norm = Standardizer.fit(x)
    recovered = norm.inverse(norm.transform(x))
    np.testing.assert_allclose(x, recovered, atol=1e-5)


def test_world_model_requires_actions() -> None:
    encoder = ObservationEncoder(10, 4, 16)
    wm = RepresentationWorldModel(latent_dim=4, action_dim=2, hidden_dim=16)
    x = torch.randn(3, 10)
    z = encoder(x)
    action_seq = torch.randn(3, 8, 2)
    horizons = torch.tensor([1, 5, 8])
    out = wm(z, action_seq, horizons)
    assert out.shape == (3, 4)


def test_flow_shapes_and_sample() -> None:
    model = FlowModel(sample_dim=6, cond_dim=4, hidden_dim=16)
    x = torch.randn(5, 6)
    cond = torch.randn(5, 4)
    loss = flow_matching_loss(model, x, cond)
    assert loss.ndim == 0
    sample = sample_flow(model, cond, steps=2, sample_dim=6)
    assert sample.shape == (5, 6)

