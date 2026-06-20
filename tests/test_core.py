from __future__ import annotations

import numpy as np
import torch

from hcl_poc.config import Config
from hcl_poc.eval import horizon_steps
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.incremental import (
    _pre_rl_phase_b_goal,
    action_alignment_metrics,
    copied_actor_student,
)
from hcl_poc.models import FlowModel, ObservationEncoder, RepresentationWorldModel
from hcl_poc.rl import PPOAgent
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
    initial_noise = torch.zeros(5, 6)
    sample_from_noise = sample_flow(model, cond, steps=2, sample_dim=6, initial_noise=initial_noise)
    assert sample_from_noise.shape == (5, 6)


def test_ppo_agent_shapes() -> None:
    agent = PPOAgent(obs_dim=31, action_dim=3, hidden_dim=16)
    obs = torch.randn(4, 31)
    action, logprob, entropy, value = agent.get_action_and_value(obs, deterministic=True)
    assert action.shape == (4, 3)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)


def test_copied_actor_student_is_exact() -> None:
    teacher = PPOAgent(obs_dim=31, action_dim=3, hidden_dim=16)
    student = copied_actor_student(teacher, torch.device("cpu"))
    obs = torch.randn(32, 31)
    torch.testing.assert_close(teacher.actor_mean(obs), student(obs), rtol=0, atol=0)


def test_action_alignment_favors_unshifted_actions() -> None:
    observations = np.zeros((5, 2), dtype=np.float32)
    teacher_actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    stored_actions = teacher_actions.copy()
    metrics = action_alignment_metrics(observations, stored_actions, teacher_actions)
    assert metrics["shift_0_mae"] == 0.0
    assert metrics["shift_0_mae"] < metrics["shift_minus_1_mae"]
    assert metrics["shift_0_mae"] < metrics["shift_plus_1_mae"]


def test_pre_rl_phase_b_goal_decomposition_has_disjoint_expected_slices() -> None:
    current = np.zeros(31, dtype=np.float32)
    future = np.zeros(31, dtype=np.float32)
    future[:14] = np.arange(14, dtype=np.float32)
    future[14:17] = [1.0, 2.0, 3.0]
    future[24:26] = [4.0, 5.0]
    yaw = 0.6
    future[27] = np.cos(yaw / 2.0)
    future[30] = np.sin(yaw / 2.0)

    object_pose = _pre_rl_phase_b_goal(current, future, 2, 20, "object_pose")
    object_goal = _pre_rl_phase_b_goal(current, future, 2, 20, "object")
    tcp_goal = _pre_rl_phase_b_goal(current, future, 2, 20, "tcp")
    robot_goal = _pre_rl_phase_b_goal(current, future, 2, 20, "robot")
    full_goal = _pre_rl_phase_b_goal(current, future, 2, 20, "full")

    assert object_pose.shape == (4,)
    assert object_goal.shape == (7,)
    assert tcp_goal.shape == (6,)
    assert robot_goal.shape == (20,)
    assert full_goal.shape == (28,)
    np.testing.assert_allclose(object_goal[:4], object_pose)
    np.testing.assert_allclose(robot_goal[:6], tcp_goal)
    np.testing.assert_allclose(robot_goal[6:], future[:14])
    np.testing.assert_allclose(full_goal[:7], object_goal)
    np.testing.assert_allclose(full_goal[7:13], tcp_goal)
    np.testing.assert_allclose(full_goal[13:27], future[:14])
    assert full_goal[-1] == 0.0
