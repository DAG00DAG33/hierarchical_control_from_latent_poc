from __future__ import annotations

import h5py
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
from hcl_poc.learned_interface import (
    _EffectEncoder,
    _EffectRepresentationDataset,
    _GoalConditionedLowPolicy,
    _HeldGoalDataset,
    _PredictiveRepresentationDataset,
    _variance_covariance_losses,
    _vae_kl,
)
from hcl_poc.models import FlowModel, ObservationEncoder, RepresentationWorldModel
from hcl_poc.rl import PPOAgent
from hcl_poc.utils import Standardizer
from hcl_poc.vae_scaling import (
    _FlatDataset,
    _dataset_content_sha256,
    _deterministic_noise,
    _threshold_crossing,
    vae_scaling_config,
)


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


def test_vae_free_bits_apply_per_dimension() -> None:
    mean = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    total, per_dimension = _vae_kl(mean, logvar, free_bits=0.01)
    torch.testing.assert_close(total, torch.tensor(0.04))
    torch.testing.assert_close(per_dimension, torch.tensor(0.0))


def test_held_goal_dataset_keeps_fixed_future_goal() -> None:
    frames = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    goals = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    actions = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    frame_norm = Standardizer.fit(frames)
    goal_norm = Standardizer.fit(goals)
    action_norm = Standardizer.fit(actions)
    dataset = _HeldGoalDataset(
        [{"frames": frames, "goals": goals, "actions": actions}],
        frame_norm,
        goal_norm,
        action_norm,
        horizon_steps=10,
        mode="low",
        length=1,
    )
    condition, target = dataset[0]
    assert condition.shape == (2 + 3 + 3 + 1,)
    assert target.shape == (3,)
    assert 0.1 <= float(condition[-1]) <= 1.0


def test_goal_conditioning_variants_have_consistent_policy_outputs() -> None:
    frames = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    goals = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    actions = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    norms = (
        Standardizer.fit(frames),
        Standardizer.fit(goals),
        Standardizer.fit(actions),
    )
    expected_dims = {"concat": 9, "delta": 9, "relation": 12, "film": 9}
    for conditioning, expected_dim in expected_dims.items():
        dataset = _HeldGoalDataset(
            [{"frames": frames, "goals": goals, "actions": actions}],
            *norms,
            horizon_steps=10,
            mode="low",
            length=1,
            conditioning=conditioning,
        )
        condition, _target = dataset[0]
        assert condition.shape == (expected_dim,)
        policy = _GoalConditionedLowPolicy(
            frame_dim=2,
            goal_dim=3,
            hidden_dim=16,
            conditioning=conditioning,
        )
        assert policy(condition[None]).shape == (1, 3)


def test_predictive_dataset_and_regularizers() -> None:
    frames = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    actions = np.arange(11 * 3, dtype=np.float32).reshape(11, 3)
    dataset = _PredictiveRepresentationDataset(
        [{"frames": frames, "actions": actions}],
        horizons=[1, 2, 5, 10],
        length=1,
    )
    sample = dataset[0]
    assert sample["x_t"].shape == (4,)
    assert sample["x_future"].shape == (4,)
    assert sample["actions"].shape == (10, 3)
    variance, covariance = _variance_covariance_losses(torch.randn(32, 8))
    assert variance.ndim == 0
    assert covariance.ndim == 0
    assert variance >= 0
    assert covariance >= 0


def test_effect_dataset_uses_one_fixed_pair_for_held_goal() -> None:
    frames = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    actions = np.arange(11 * 3, dtype=np.float32).reshape(11, 3)
    auxiliary = np.arange(12 * 12, dtype=np.float32).reshape(12, 12)
    dataset = _EffectRepresentationDataset(
        [
            {
                "frames": frames,
                "actions": actions,
                "zero_action": np.zeros(3, dtype=np.float32),
                "auxiliary": auxiliary,
            }
        ],
        horizon_steps=10,
        effect_input_dim=4,
        length=1,
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["x_start"], frames[0])
    np.testing.assert_array_equal(sample["x_future"], frames[10])
    np.testing.assert_array_equal(sample["auxiliary"], auxiliary[10])
    assert 0.1 <= float(sample["remaining"]) <= 1.0
    encoder = _EffectEncoder(input_dim=4, effect_dim=8, hidden_dim=16)
    pair = torch.cat(
        [sample["x_start"], sample["x_future"], torch.ones(1)]
    )[None]
    assert encoder(pair).shape == (1, 8)


def test_vae_scaling_budget_config_is_isolated() -> None:
    config = Config(
        raw={
            "paths": {
                "incremental_artifact_dir": "artifacts/incremental",
                "incremental_results_dir": "results/incremental",
            },
            "incremental": {
                "phase4": {"train_episodes": 1800},
                "phase6": {"train_episodes": 1800},
            },
            "learned_interface": {"evaluation": {"seed_start": 1}},
            "vae_scaling": {
                "eval_seed_start": 2200000,
                "extended_prepared_path": "data/extended.h5",
            },
        },
        path="dummy.yaml",  # type: ignore[arg-type]
    )
    point = vae_scaling_config(config, 100)
    assert point.get("incremental.phase4.train_episodes") == 100
    assert point.get("incremental.phase6.train_episodes") == 100
    assert point.get("learned_interface.evaluation.seed_start") == 2200000
    assert str(point.get("paths.incremental_artifact_dir")).endswith(
        "vae512_scaling/n100"
    )
    assert config.get("incremental.phase4.train_episodes") == 1800
    extended = vae_scaling_config(config, 4000)
    assert extended.get("incremental.phase4.train_episodes") == 4000
    assert extended.get("incremental.phase4.prepared_path") == "data/extended.h5"


def test_dataset_content_hash_ignores_episode_group_names(tmp_path) -> None:
    paths = [tmp_path / "first.h5", tmp_path / "second.h5"]
    names = ["episode_1800", "episode_8000"]
    for path, name in zip(paths, names, strict=True):
        with h5py.File(path, "w") as h5:
            group = h5.create_group(name)
            group.create_dataset("dino", data=np.arange(12).reshape(3, 4))
            group.create_dataset("proprio", data=np.arange(6).reshape(3, 2))
            group.create_dataset("actions", data=np.arange(9).reshape(3, 3))
    hashes = []
    for path, name in zip(paths, names, strict=True):
        with h5py.File(path, "r") as h5:
            hashes.append(_dataset_content_sha256(h5, [name]))
    assert hashes[0] == hashes[1]


def test_vae_scaling_flow_noise_is_reproducible_and_seed_specific() -> None:
    first = _deterministic_noise(0, [11, 12], 3, 8)
    repeated = _deterministic_noise(0, [11, 12], 3, 8)
    different_policy = _deterministic_noise(1, [11, 12], 3, 8)
    different_decision = _deterministic_noise(0, [11, 12], 4, 8)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different_policy)
    assert not np.array_equal(first, different_decision)


def test_vae_scaling_threshold_crossing_interpolates_in_log_budget() -> None:
    budgets = np.asarray([50, 100, 200], dtype=float)
    values = np.asarray([0.2, 0.4, 0.6], dtype=float)
    assert np.isclose(_threshold_crossing(budgets, values, 0.5), np.sqrt(20_000))
    assert _threshold_crossing(budgets, values, 0.7) is None


def test_vae_scaling_flat_dataset_uses_previous_executed_action() -> None:
    frames = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    actions = np.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32)
    dataset = _FlatDataset(
        [{"frames": frames, "latents": frames, "actions": actions}],
        "frames",
        Standardizer.fit(frames),
        Standardizer.fit(actions),
        length=1,
    )
    condition, target = dataset[0]
    assert condition.shape == (7,)
    assert target.shape == (3,)


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
