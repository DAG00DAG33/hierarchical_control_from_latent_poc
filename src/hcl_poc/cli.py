from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import torch
from rich.console import Console

from hcl_poc.config import load_config
from hcl_poc.data import prepare_dataset
from hcl_poc.eval import evaluate, horizon_steps, record_videos
from hcl_poc.incremental import (
    collect_phase1_query_dataset,
    collect_phase2_dagger_queries,
    collect_phase6_latent_dagger_queries,
    collect_phase7_oracle_dagger_queries,
    collect_phase8_dagger_queries,
    collect_phase10_flow_queries,
    evaluate_phase1_bc,
    evaluate_phase2_dagger_bc,
    evaluate_phase2_recovery,
    evaluate_phase3_flow,
    evaluate_phase4_visual_bc,
    evaluate_phase5_visual_flow,
    evaluate_phase6_latent_bc,
    evaluate_phase6_latent_dagger_bc,
    evaluate_phase6_latent_flow,
    evaluate_phase7_matched_flat_latent_policy,
    evaluate_phase7_oracle_low_level,
    evaluate_phase7_oracle_dagger_low_level,
    evaluate_phase7_privileged_branch_baselines,
    evaluate_phase7_replay_branch_oracle_low_level,
    evaluate_phase7_valid_goal_use,
    evaluate_phase8_deterministic_hierarchy,
    evaluate_phase8_structured_hierarchy,
    evaluate_phase9_future_flow,
    probe_phase6_representation,
    probe_phase4_visual_history,
    prepare_phase8_latent_episodes,
    probe_phase8_predicted_latents,
    run_phase0,
    run_phase7_branch_audit,
    run_phase11_comparison,
    run_phase12_budget,
    plot_phase12_sample_efficiency,
    run_pre_rl_phase_a_seed,
    aggregate_pre_rl_phase_a,
    train_pre_rl_phase_b_horizon,
    evaluate_pre_rl_phase_b_horizon,
    aggregate_pre_rl_phase_b,
    run_pre_rl_phase_c_oracle_sweep,
    train_pre_rl_phase_c_time_conditioned,
    collect_pre_rl_phase_d_recovery_dataset,
    prepare_pre_rl_phase_d_features,
    create_pre_rl_phase_d_manifests,
    train_pre_rl_phase_d_visual_bc,
    evaluate_pre_rl_phase_d_visual_bc,
    analyze_pre_rl_phase_e_geometry,
    train_pre_rl_phase_f_privileged_tcp_predictor,
    evaluate_pre_rl_phase_f_privileged_tcp_hierarchy,
    train_pre_rl_phase_f_visual_tcp_hierarchy,
    evaluate_pre_rl_phase_f_visual_tcp_hierarchy,
    record_pre_rl_phase_f_visual_tcp_videos,
    create_pre_rl_phase_d_hierarchy_manifests,
    train_pre_rl_phase_d_raw_tcp_hierarchy,
    evaluate_pre_rl_phase_d_raw_tcp_hierarchy,
    analyze_pre_rl_phase_g_tcp_predictor,
    train_phase1_bc,
    train_phase2_dagger_bc,
    train_phase3_flow,
    train_phase4_visual_bc,
    train_phase5_visual_flow,
    train_phase6_latent_bc,
    train_phase6_latent_dagger_bc,
    train_phase6_latent_flow,
    train_phase6_representation,
    train_phase7_oracle_low_level,
    train_phase7_oracle_dagger_low_level,
    train_phase7_privileged_branch_baselines,
    train_phase7_residual_low_level,
    train_phase8_deterministic_predictor,
    train_phase8_dagger_predictor,
    train_phase8_adapted_low_level,
    train_phase8_action_consistent_predictor,
    train_phase8_structured_predictor,
    train_phase9_future_flow,
    train_phase10_robust_low_level,
    sweep_phase8_deterministic_predictors,
)
from hcl_poc.learned_interface import (
    evaluate_learned_interface_hierarchy,
    prepare_learned_interface_episodes,
    probe_learned_interface_representation,
    record_learned_interface_videos,
    run_learned_interface_candidate,
    train_learned_interface_hierarchy,
    train_learned_interface_representation,
)
from hcl_poc.low_level_rl import (
    audit_low_level_rl,
    evaluate_residual_rl,
    train_direct_low_rl,
    train_residual_rl,
)
from hcl_poc.report import build_report
from hcl_poc.rl import collect_ppo_dataset, evaluate_ppo, ppo_status, train_ppo
from hcl_poc.train import (
    diagnose_hierarchy,
    train_bc_policy,
    train_dagger_bc_policy,
    train_flow_policy,
    probe_latent_pose,
    train_pose_bc_policy,
    train_representation,
    train_state_bc_policy,
)
from hcl_poc.vae_scaling import (
    aggregate_vae_scaling_results,
    evaluate_vae_scaling_point,
    extend_vae_scaling_dataset,
    train_vae_scaling_point,
    validate_nested_vae_scaling_manifests,
)

console = Console()


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pusht.yaml")


def low_level_rl_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.low_level_rl_command == "audit":
        console.print(audit_low_level_rl(config, args.n_demo, args.seed))
    elif args.low_level_rl_command == "train-r1":
        run_config = config
        if args.no_segment_terminate_gae:
            raw = copy.deepcopy(config.raw)
            raw.setdefault("low_level_rl", {})["segment_terminates_gae"] = False
            run_config = type(config)(raw=raw, path=config.path)
        console.print(
            train_residual_rl(
                run_config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                alpha=args.alpha,
                terminal_weight=args.terminal_weight,
                task_reward_weight=args.task_reward_weight,
                task_progress_weight=args.task_progress_weight,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "train-r3":
        run_config = config
        if args.no_segment_terminate_gae:
            raw = copy.deepcopy(config.raw)
            raw.setdefault("low_level_rl", {})["segment_terminates_gae"] = False
            run_config = type(config)(raw=raw, path=config.path)
        console.print(
            train_direct_low_rl(
                run_config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                bc_weight=args.bc_weight,
                terminal_weight=args.terminal_weight,
                task_reward_weight=args.task_reward_weight,
                task_progress_weight=args.task_progress_weight,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "eval":
        console.print(
            evaluate_residual_rl(
                config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                episodes=args.episodes,
                seed_start=args.seed_start,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                force=args.force,
            )
        )
    else:
        raise ValueError(args.low_level_rl_command)


def doctor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        console.print(f"GPU memory free/total: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    try:
        import gymnasium as gym
        import mani_skill  # noqa: F401

        env = gym.make(
            config.get("env_id"),
            obs_mode=config.get("obs_mode"),
            control_mode=config.get("control_mode"),
        )
        console.print(f"ManiSkill env OK: {config.get('env_id')}")
        console.print(f"Action space: {env.action_space}")
        env.close()
    except Exception as exc:
        raise RuntimeError("ManiSkill environment check failed") from exc


def data_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.data_command == "prepare":
        if config.get("data.source") == "privileged_ppo":
            collect_ppo_dataset(config, force=args.force)
        else:
            prepare_dataset(config, force=args.force)
    else:
        raise ValueError(args.data_command)


def train_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.kind == "encoder":
        train_representation(config, args.n_traj, args.seed)
    elif args.kind in {"flat", "flat_obs"}:
        train_flow_policy(config, args.n_traj, args.seed, args.kind, force=args.force)
    elif args.kind == "bc_obs":
        train_bc_policy(config, args.n_traj, args.seed, force=args.force)
    elif args.kind == "bc_obs_1step":
        train_bc_policy(config, args.n_traj, args.seed, force=args.force, one_step=True)
    elif args.kind == "bc_obs_dagger":
        train_dagger_bc_policy(config, args.n_traj, args.seed, force=args.force)
    elif args.kind == "bc_pose":
        train_pose_bc_policy(config, args.n_traj, args.seed, force=args.force)
    elif args.kind == "bc_state":
        train_state_bc_policy(config, args.n_traj, args.seed, force=args.force)
    elif args.kind in {"high", "low"}:
        if args.horizon_s is None:
            raise ValueError("--horizon-s is required for high/low")
        train_flow_policy(
            config,
            args.n_traj,
            args.seed,
            args.kind,
            horizon_steps(config, args.horizon_s),
            force=args.force,
        )
    else:
        raise ValueError(args.kind)


def eval_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    evaluate(config, args.n_traj, args.seed, args.method, args.horizon_s, episodes=args.episodes)


def video_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    record_videos(config, args.n_traj, args.seed, args.method, args.episodes, args.horizon_s)


def run_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    profile = args.profile
    seeds = [0] if profile == "staged" else [0, 1, 2]
    n_values = [int(n) for n in config.get("data.train_trajectories")]
    horizons = [float(h) for h in config.get("policy.high_level_horizons_s")]

    if config.get("data.source") == "privileged_ppo":
        collect_ppo_dataset(config, force=False)
    else:
        prepare_dataset(config, force=False)
    for seed in seeds:
        for n_traj in n_values:
            train_representation(config, n_traj, seed)
            train_flow_policy(config, n_traj, seed, "flat")
            evaluate(config, n_traj, seed, "flat")
            for horizon_s in horizons:
                h_steps = horizon_steps(config, horizon_s)
                train_flow_policy(config, n_traj, seed, "high", h_steps)
                train_flow_policy(config, n_traj, seed, "low", h_steps)
                evaluate(config, n_traj, seed, "hier", horizon_s)
    build_report(config)


def report_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = build_report(config)
    console.print(f"Wrote {path}")


def probe_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    probe_latent_pose(
        config,
        args.n_traj,
        args.seed,
        Path(args.samples_file),
        Path(args.out),
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
    )


def diagnose_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    diagnose_hierarchy(
        config,
        args.n_traj,
        args.seed,
        args.horizon_s,
        args.samples,
        Path(args.out),
    )


def rl_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.rl_command == "train":
        train_ppo(config, resume=not args.no_resume)
    elif args.rl_command == "status":
        ppo_status(config)
    elif args.rl_command == "eval":
        evaluate_ppo(config, checkpoint=args.checkpoint, episodes=args.episodes)
    elif args.rl_command == "collect":
        collect_ppo_dataset(
            config,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            force=args.force,
        )
    else:
        raise ValueError(args.rl_command)


def incremental_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.incremental_command == "phase0":
        run_phase0(config, episodes=args.episodes, force=args.force)
    elif args.incremental_command == "phase1-collect":
        collect_phase1_query_dataset(config, force=args.force)
    elif args.incremental_command == "phase1-train":
        train_phase1_bc(
            config,
            n_episodes=args.n_episodes,
            seed=args.seed,
            subset=args.subset,
            label_kind=args.label_kind,
            force=args.force,
        )
    elif args.incremental_command == "phase1-eval":
        evaluate_phase1_bc(
            config,
            n_episodes=args.n_episodes,
            seed=args.seed,
            subset=args.subset,
            label_kind=args.label_kind,
            episodes=args.episodes,
        )
    elif args.incremental_command == "phase2-collect":
        collect_phase2_dagger_queries(
            config,
            iteration=args.iteration,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase2-train":
        train_phase2_dagger_bc(
            config,
            iteration=args.iteration,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase2-eval":
        evaluate_phase2_dagger_bc(
            config,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
        )
    elif args.incremental_command == "phase2-recovery":
        evaluate_phase2_recovery(
            config,
            iteration=args.iteration,
            seed=args.seed,
            samples=args.samples,
            force=args.force,
        )
    elif args.incremental_command == "phase3-train":
        train_phase3_flow(config, seed=args.seed, force=args.force)
    elif args.incremental_command == "phase3-eval":
        evaluate_phase3_flow(config, seed=args.seed, episodes=args.episodes)
    elif args.incremental_command == "phase4-train":
        train_phase4_visual_bc(
            config,
            history=args.history,
            architecture=args.architecture,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase4-eval":
        evaluate_phase4_visual_bc(
            config,
            history=args.history,
            architecture=args.architecture,
            seed=args.seed,
            episodes=args.episodes,
        )
    elif args.incremental_command == "phase4-probe":
        probe_phase4_visual_history(
            config,
            history=args.history,
            architecture=args.architecture,
            seed=args.seed,
            samples=args.samples,
            force=args.force,
        )
    elif args.incremental_command == "phase5-train":
        train_phase5_visual_flow(
            config,
            history=args.history,
            architecture=args.architecture,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase5-eval":
        evaluate_phase5_visual_flow(
            config,
            history=args.history,
            architecture=args.architecture,
            seed=args.seed,
            episodes=args.episodes,
        )
    elif args.incremental_command == "phase6-train":
        train_phase6_representation(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase6-probe":
        probe_phase6_representation(
            config,
            representation=args.representation,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase6-control-train":
        train_phase6_latent_bc(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase6-control-eval":
        evaluate_phase6_latent_bc(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase6-flow-train":
        train_phase6_latent_flow(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase6-flow-eval":
        evaluate_phase6_latent_flow(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase6-dagger-collect":
        collect_phase6_latent_dagger_queries(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase6-dagger-train":
        train_phase6_latent_dagger_bc(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            iteration=args.iteration,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase6-dagger-eval":
        evaluate_phase6_latent_dagger_bc(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-train":
        train_phase7_oracle_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase7-eval":
        evaluate_phase7_oracle_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            episodes=args.episodes,
            goal_mode=args.goal_mode,
            force=args.force,
        )
    elif args.incremental_command == "phase7-branch-audit":
        run_phase7_branch_audit(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            trials=args.trials,
            warmup_steps=args.warmup_steps,
            force=args.force,
        )
    elif args.incremental_command == "phase7-replay-branch-eval":
        evaluate_phase7_replay_branch_oracle_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            episodes=args.episodes,
            dagger_iteration=args.dagger_iteration,
            dagger_query_episodes=args.dagger_query_episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-residual-train":
        train_phase7_residual_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase7-residual-replay-eval":
        evaluate_phase7_replay_branch_oracle_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            episodes=args.episodes,
            residual=True,
            force=args.force,
        )
    elif args.incremental_command == "phase7-matched-flat-eval":
        evaluate_phase7_matched_flat_latent_policy(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-goal-use-eval":
        evaluate_phase7_valid_goal_use(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            seed=args.seed,
            episodes=args.episodes,
            dagger_iteration=args.dagger_iteration,
            dagger_query_episodes=args.dagger_query_episodes,
            counterfactual_queries=args.counterfactual_queries,
            force=args.force,
        )
    elif args.incremental_command == "phase7-priv-train":
        train_phase7_privileged_branch_baselines(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase7-priv-eval":
        evaluate_phase7_privileged_branch_baselines(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-dagger-collect":
        collect_phase7_oracle_dagger_queries(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-dagger-train":
        train_phase7_oracle_dagger_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            iteration=args.iteration,
            seed=args.seed,
            query_episodes=args.query_episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase7-dagger-eval":
        evaluate_phase7_oracle_dagger_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            goal_encoding=args.goal_encoding,
            goal_dropout_prob=args.goal_dropout_prob,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
            query_episodes=args.query_episodes,
            goal_mode=args.goal_mode,
            force=args.force,
        )
    elif args.incremental_command == "phase8-prepare":
        prepare_phase8_latent_episodes(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-train":
        train_phase8_deterministic_predictor(
            config,
            history=args.history,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            target_mode=args.target_mode,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-structured-train":
        train_phase8_structured_predictor(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-structured-eval":
        evaluate_phase8_structured_hierarchy(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase8-probe-predictions":
        probe_phase8_predicted_latents(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-dagger-train":
        train_phase8_dagger_predictor(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            query_episodes=args.query_episodes,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-dagger-collect":
        collect_phase8_dagger_queries(
            config,
            history=args.history,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            iteration=args.iteration,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase8-low-adapt":
        train_phase8_adapted_low_level(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            query_episodes=args.query_episodes,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-action-train":
        train_phase8_action_consistent_predictor(
            config,
            action_consistency_weight=args.action_consistency_weight,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-sweep":
        sweep_phase8_deterministic_predictors(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            histories=args.histories,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase8-eval":
        evaluate_phase8_deterministic_hierarchy(
            config,
            history=args.history,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            target_mode=args.target_mode,
            seed=args.seed,
            episodes=args.episodes,
            high_dagger_query_episodes=args.high_dagger_query_episodes,
            adapted_low_query_episodes=args.adapted_low_query_episodes,
            branch_action_weight=args.branch_action_weight,
            action_consistency_weight=args.action_consistency_weight,
            force=args.force,
        )
    elif args.incremental_command == "phase9-train":
        train_phase9_future_flow(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            trajectory_limit=args.trajectory_limit,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "phase9-eval":
        evaluate_phase9_future_flow(
            config,
            sample_mode=args.sample_mode,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase10-collect":
        collect_phase10_flow_queries(
            config,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase10-train":
        train_phase10_robust_low_level(
            config,
            method=args.method,
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            interpolation_alpha=args.interpolation_alpha,
            seed=args.seed,
            query_episodes=args.query_episodes,
            force=args.force,
        )
    elif args.incremental_command == "phase10-eval":
        evaluate_phase9_future_flow(
            config,
            sample_mode="zero",
            latent_dim=args.latent_dim,
            variant=args.variant,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            robust_low_method=args.method,
            interpolation_alpha=args.interpolation_alpha,
            force=args.force,
        )
    elif args.incremental_command == "phase11-eval":
        run_phase11_comparison(
            config,
            seed=args.seed,
            episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
        )
    elif args.incremental_command == "phase12-run":
        run_phase12_budget(
            config,
            n_trajectories=args.n_trajectories,
            seed=args.seed,
            episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
        )
    elif args.incremental_command == "phase12-plot":
        plot_phase12_sample_efficiency(config)
    elif args.incremental_command == "pre-rl-a-run":
        run_pre_rl_phase_a_seed(config, seed=args.seed)
    elif args.incremental_command == "pre-rl-a-aggregate":
        aggregate_pre_rl_phase_a(config)
    elif args.incremental_command == "pre-rl-b-train":
        train_pre_rl_phase_b_horizon(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-b-eval":
        evaluate_pre_rl_phase_b_horizon(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-b-aggregate":
        aggregate_pre_rl_phase_b(config, episodes=args.episodes)
    elif args.incremental_command == "pre-rl-c-oracle-sweep":
        run_pre_rl_phase_c_oracle_sweep(
            config,
            episodes=args.episodes,
            time_conditioned=args.time_conditioned,
            horizons_override=args.horizons,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-c-train-time-conditioned":
        train_pre_rl_phase_c_time_conditioned(
            config,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-d-collect":
        collect_pre_rl_phase_d_recovery_dataset(
            config,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-d-prepare":
        prepare_pre_rl_phase_d_features(
            config,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-d-manifests":
        create_pre_rl_phase_d_manifests(config, force=args.force)
    elif args.incremental_command == "pre-rl-d-train-visual-bc":
        train_pre_rl_phase_d_visual_bc(
            config,
            variant=args.variant,
            label_view=args.label_view,
            seed=args.seed,
            force=args.force,
            matched_hierarchy_data=args.matched_hierarchy_data,
        )
    elif args.incremental_command == "pre-rl-d-eval-visual-bc":
        evaluate_pre_rl_phase_d_visual_bc(
            config,
            variant=args.variant,
            label_view=args.label_view,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
            matched_hierarchy_data=args.matched_hierarchy_data,
        )
    elif args.incremental_command == "pre-rl-e-geometry":
        analyze_pre_rl_phase_e_geometry(config)
    elif args.incremental_command == "pre-rl-f-train-privileged-tcp":
        train_pre_rl_phase_f_privileged_tcp_predictor(
            config,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-f-eval-privileged-tcp":
        evaluate_pre_rl_phase_f_privileged_tcp_hierarchy(
            config,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-f-train-visual-tcp":
        train_pre_rl_phase_f_visual_tcp_hierarchy(
            config,
            representation=args.representation,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-f-eval-visual-tcp":
        evaluate_pre_rl_phase_f_visual_tcp_hierarchy(
            config,
            representation=args.representation,
            goal_source=args.goal_source,
            seed=args.seed,
            episodes=args.episodes,
            audit_branch=args.audit_branch,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-f-record-visual-tcp":
        record_pre_rl_phase_f_visual_tcp_videos(
            config,
            representation=args.representation,
            goal_source=args.goal_source,
            seed=args.seed,
            episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-d-hierarchy-manifests":
        create_pre_rl_phase_d_hierarchy_manifests(config, force=args.force)
    elif args.incremental_command == "pre-rl-d-train-hierarchy":
        train_pre_rl_phase_d_raw_tcp_hierarchy(
            config,
            variant=args.variant,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-d-eval-hierarchy":
        evaluate_pre_rl_phase_d_raw_tcp_hierarchy(
            config,
            variant=args.variant,
            disturbed=args.disturbed,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "pre-rl-g-tcp-diagnostics":
        analyze_pre_rl_phase_g_tcp_predictor(config, force=args.force)
    elif args.incremental_command == "learned-interface-train-representation":
        train_learned_interface_representation(
            config,
            candidate=args.candidate,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-probe":
        probe_learned_interface_representation(
            config,
            candidate=args.candidate,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-prepare":
        prepare_learned_interface_episodes(
            config,
            candidate=args.candidate,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-train-hierarchy":
        train_learned_interface_hierarchy(
            config,
            candidate=args.candidate,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-eval":
        evaluate_learned_interface_hierarchy(
            config,
            candidate=args.candidate,
            goal_source=args.goal_source,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-run":
        run_learned_interface_candidate(
            config,
            candidate=args.candidate,
            seed=args.seed,
            episodes=args.episodes,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-record":
        record_learned_interface_videos(
            config,
            candidate=args.candidate,
            goal_source=args.goal_source,
            seed=args.seed,
            episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
            force=args.force,
        )
    elif args.incremental_command == "vae-scaling-manifests":
        console.print(validate_nested_vae_scaling_manifests(config))
    elif args.incremental_command == "vae-scaling-extend-data":
        console.print(extend_vae_scaling_dataset(config, force=args.force))
    elif args.incremental_command == "vae-scaling-aggregate":
        console.print(
            aggregate_vae_scaling_results(
                config,
                deployable_episodes=args.episodes,
                oracle_episodes=args.oracle_episodes,
                training_seeds=tuple(args.seeds),
                output_name=args.output_name,
            )
        )
    elif args.incremental_command == "vae-scaling-train":
        train_vae_scaling_point(
            config,
            n_trajectories=args.n_trajectories,
            seed=args.seed,
            force=args.force,
        )
    elif args.incremental_command in {"vae-scaling-eval", "vae-scaling-run"}:
        evaluate_vae_scaling_point(
            config,
            n_trajectories=args.n_trajectories,
            seed=args.seed,
            deployable_episodes=args.episodes,
            oracle_episodes=args.oracle_episodes,
            force=args.force,
        )
    else:
        raise ValueError(args.incremental_command)


def commit_cmd(args: argparse.Namespace) -> None:
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", args.message], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcl-poc")
    parser.add_argument("--config", default="configs/pusht.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor")
    add_config_arg(p)
    p.set_defaults(func=doctor)

    p = sub.add_parser("data")
    data_sub = p.add_subparsers(dest="data_command", required=True)
    pp = data_sub.add_parser("prepare")
    add_config_arg(pp)
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=data_cmd)

    p = sub.add_parser("low-level-rl")
    add_config_arg(p)
    low_level_rl_sub = p.add_subparsers(dest="low_level_rl_command", required=True)
    audit = low_level_rl_sub.add_parser("audit")
    audit.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    audit.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    audit.set_defaults(func=low_level_rl_cmd)
    train_r1 = low_level_rl_sub.add_parser("train-r1")
    train_r1.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    train_r1.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    train_r1.add_argument("--run-name", required=True)
    train_r1.add_argument("--steps", type=int, required=True)
    train_r1.add_argument("--alpha", type=float, default=0.1)
    train_r1.add_argument("--terminal-weight", type=float, default=1.0)
    train_r1.add_argument("--task-reward-weight", type=float, default=0.0)
    train_r1.add_argument("--task-progress-weight", type=float, default=0.0)
    train_r1.add_argument("--no-segment-terminate-gae", action="store_true")
    train_r1.add_argument("--force", action="store_true")
    train_r1.set_defaults(func=low_level_rl_cmd)
    train_r3 = low_level_rl_sub.add_parser("train-r3")
    train_r3.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    train_r3.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    train_r3.add_argument("--run-name", required=True)
    train_r3.add_argument("--steps", type=int, required=True)
    train_r3.add_argument("--bc-weight", type=float, default=1.0)
    train_r3.add_argument("--terminal-weight", type=float, default=1.0)
    train_r3.add_argument("--task-reward-weight", type=float, default=0.0)
    train_r3.add_argument("--task-progress-weight", type=float, default=0.0)
    train_r3.add_argument("--no-segment-terminate-gae", action="store_true")
    train_r3.add_argument("--force", action="store_true")
    train_r3.set_defaults(func=low_level_rl_cmd)
    low_eval = low_level_rl_sub.add_parser("eval")
    low_eval.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    low_eval.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    low_eval.add_argument("--run-name", required=True)
    low_eval.add_argument("--episodes", type=int, required=True)
    low_eval.add_argument("--seed-start", type=int, required=True)
    low_eval.add_argument("--checkpoint")
    low_eval.add_argument("--force", action="store_true")
    low_eval.set_defaults(func=low_level_rl_cmd)

    p = sub.add_parser("rl")
    add_config_arg(p)
    rl_sub = p.add_subparsers(dest="rl_command", required=True)
    rt = rl_sub.add_parser("train")
    add_config_arg(rt)
    rt.add_argument("--no-resume", action="store_true")
    rt.set_defaults(func=rl_cmd)
    rs = rl_sub.add_parser("status")
    add_config_arg(rs)
    rs.set_defaults(func=rl_cmd)
    re = rl_sub.add_parser("eval")
    add_config_arg(re)
    re.add_argument("--checkpoint")
    re.add_argument("--episodes", type=int)
    re.set_defaults(func=rl_cmd)
    rc = rl_sub.add_parser("collect")
    add_config_arg(rc)
    rc.add_argument("--checkpoint")
    rc.add_argument("--episodes", type=int)
    rc.add_argument("--force", action="store_true")
    rc.set_defaults(func=rl_cmd)

    p = sub.add_parser("incremental")
    incremental_sub = p.add_subparsers(dest="incremental_command", required=True)
    phase0 = incremental_sub.add_parser("phase0")
    add_config_arg(phase0)
    phase0.add_argument("--episodes", type=int)
    phase0.add_argument("--force", action="store_true")
    phase0.set_defaults(func=incremental_cmd)
    phase1_collect = incremental_sub.add_parser("phase1-collect")
    add_config_arg(phase1_collect)
    phase1_collect.add_argument("--force", action="store_true")
    phase1_collect.set_defaults(func=incremental_cmd)
    for command in ["phase1-train", "phase1-eval"]:
        phase1 = incremental_sub.add_parser(command)
        add_config_arg(phase1)
        phase1.add_argument("--n-episodes", type=int)
        phase1.add_argument("--seed", type=int, default=0)
        phase1.add_argument("--subset", choices=["all", "successful"], default="all")
        phase1.add_argument(
            "--label-kind",
            choices=["deterministic_clipped", "deterministic_raw"],
            default="deterministic_clipped",
        )
        phase1.add_argument("--force", action="store_true")
        if command == "phase1-eval":
            phase1.add_argument("--episodes", type=int)
        phase1.set_defaults(func=incremental_cmd)
    for command in ["phase2-collect", "phase2-train", "phase2-eval"]:
        phase2 = incremental_sub.add_parser(command)
        add_config_arg(phase2)
        phase2.add_argument("--iteration", type=int, required=True)
        phase2.add_argument("--seed", type=int, default=0)
        phase2.add_argument("--force", action="store_true")
        if command == "phase2-eval":
            phase2.add_argument("--episodes", type=int)
        phase2.set_defaults(func=incremental_cmd)
    phase2_recovery = incremental_sub.add_parser("phase2-recovery")
    add_config_arg(phase2_recovery)
    phase2_recovery.add_argument("--iteration", type=int, default=3)
    phase2_recovery.add_argument("--seed", type=int, default=0)
    phase2_recovery.add_argument("--samples", type=int)
    phase2_recovery.add_argument("--force", action="store_true")
    phase2_recovery.set_defaults(func=incremental_cmd)
    for command in ["phase3-train", "phase3-eval"]:
        phase3 = incremental_sub.add_parser(command)
        add_config_arg(phase3)
        phase3.add_argument("--seed", type=int, default=0)
        phase3.add_argument("--force", action="store_true")
        if command == "phase3-eval":
            phase3.add_argument("--episodes", type=int)
        phase3.set_defaults(func=incremental_cmd)
    for command in ["phase4-train", "phase4-eval", "phase4-probe"]:
        phase4 = incremental_sub.add_parser(command)
        add_config_arg(phase4)
        phase4.add_argument("--history", type=int, required=True)
        phase4.add_argument("--architecture", default=None)
        phase4.add_argument("--seed", type=int, default=0)
        phase4.add_argument("--force", action="store_true")
        if command == "phase4-eval":
            phase4.add_argument("--episodes", type=int)
        if command == "phase4-probe":
            phase4.add_argument("--samples", type=int)
        phase4.set_defaults(func=incremental_cmd)
    for command in ["phase5-train", "phase5-eval"]:
        phase5 = incremental_sub.add_parser(command)
        add_config_arg(phase5)
        phase5.add_argument("--history", type=int)
        phase5.add_argument("--architecture", default=None)
        phase5.add_argument("--seed", type=int, default=0)
        phase5.add_argument("--force", action="store_true")
        if command == "phase5-eval":
            phase5.add_argument("--episodes", type=int)
        phase5.set_defaults(func=incremental_cmd)
    phase6_train = incremental_sub.add_parser("phase6-train")
    add_config_arg(phase6_train)
    phase6_train.add_argument("--latent-dim", type=int, required=True)
    phase6_train.add_argument("--variant", default=None)
    phase6_train.add_argument("--seed", type=int, default=0)
    phase6_train.add_argument("--force", action="store_true")
    phase6_train.set_defaults(func=incremental_cmd)
    phase6_probe = incremental_sub.add_parser("phase6-probe")
    add_config_arg(phase6_probe)
    phase6_probe.add_argument("--representation", choices=["raw", "latent"], default="raw")
    phase6_probe.add_argument("--latent-dim", type=int)
    phase6_probe.add_argument("--variant", default=None)
    phase6_probe.add_argument("--seed", type=int, default=0)
    phase6_probe.add_argument("--force", action="store_true")
    phase6_probe.set_defaults(func=incremental_cmd)
    for command in ["phase6-control-train", "phase6-control-eval"]:
        phase6_control = incremental_sub.add_parser(command)
        add_config_arg(phase6_control)
        phase6_control.add_argument("--latent-dim", type=int, required=True)
        phase6_control.add_argument("--variant", default=None)
        phase6_control.add_argument("--seed", type=int, default=0)
        phase6_control.add_argument("--force", action="store_true")
        if command == "phase6-control-eval":
            phase6_control.add_argument("--episodes", type=int)
        phase6_control.set_defaults(func=incremental_cmd)
    for command in ["phase6-flow-train", "phase6-flow-eval"]:
        phase6_flow = incremental_sub.add_parser(command)
        add_config_arg(phase6_flow)
        phase6_flow.add_argument("--latent-dim", type=int, required=True)
        phase6_flow.add_argument("--variant", default=None)
        phase6_flow.add_argument("--seed", type=int, default=0)
        phase6_flow.add_argument("--force", action="store_true")
        if command == "phase6-flow-eval":
            phase6_flow.add_argument("--episodes", type=int)
        phase6_flow.set_defaults(func=incremental_cmd)
    for command in ["phase6-dagger-collect", "phase6-dagger-train", "phase6-dagger-eval"]:
        phase6_dagger = incremental_sub.add_parser(command)
        add_config_arg(phase6_dagger)
        phase6_dagger.add_argument("--latent-dim", type=int, required=True)
        phase6_dagger.add_argument("--variant", default=None)
        phase6_dagger.add_argument("--iteration", type=int, default=1)
        phase6_dagger.add_argument("--seed", type=int, default=0)
        phase6_dagger.add_argument("--force", action="store_true")
        if command in {"phase6-dagger-collect", "phase6-dagger-eval"}:
            phase6_dagger.add_argument("--episodes", type=int)
        phase6_dagger.set_defaults(func=incremental_cmd)
    for command in ["phase7-train", "phase7-eval"]:
        phase7 = incremental_sub.add_parser(command)
        add_config_arg(phase7)
        phase7.add_argument("--latent-dim", type=int)
        phase7.add_argument("--variant", default=None)
        phase7.add_argument("--horizon-steps", type=int)
        phase7.add_argument("--action-chunk-steps", type=int)
        phase7.add_argument("--goal-encoding", choices=["absolute", "delta"], default=None)
        phase7.add_argument("--goal-dropout-prob", type=float)
        phase7.add_argument("--seed", type=int, default=0)
        phase7.add_argument("--force", action="store_true")
        if command == "phase7-eval":
            phase7.add_argument("--episodes", type=int)
            phase7.add_argument(
                "--goal-mode", choices=["all", "correct", "shuffled", "zero"], default="all"
            )
        phase7.set_defaults(func=incremental_cmd)
    phase7_branch = incremental_sub.add_parser("phase7-branch-audit")
    add_config_arg(phase7_branch)
    phase7_branch.add_argument("--latent-dim", type=int)
    phase7_branch.add_argument("--variant", default=None)
    phase7_branch.add_argument("--seed", type=int, default=0)
    phase7_branch.add_argument("--trials", type=int)
    phase7_branch.add_argument("--warmup-steps", type=int)
    phase7_branch.add_argument("--force", action="store_true")
    phase7_branch.set_defaults(func=incremental_cmd)
    phase7_replay = incremental_sub.add_parser("phase7-replay-branch-eval")
    add_config_arg(phase7_replay)
    phase7_replay.add_argument("--latent-dim", type=int)
    phase7_replay.add_argument("--variant", default=None)
    phase7_replay.add_argument("--horizon-steps", type=int)
    phase7_replay.add_argument("--action-chunk-steps", type=int)
    phase7_replay.add_argument("--goal-encoding", choices=["absolute", "delta"], default=None)
    phase7_replay.add_argument("--goal-dropout-prob", type=float)
    phase7_replay.add_argument("--seed", type=int, default=0)
    phase7_replay.add_argument("--episodes", type=int)
    phase7_replay.add_argument("--dagger-iteration", type=int)
    phase7_replay.add_argument("--dagger-query-episodes", type=int)
    phase7_replay.add_argument("--force", action="store_true")
    phase7_replay.set_defaults(func=incremental_cmd)
    for command in ["phase7-residual-train", "phase7-residual-replay-eval"]:
        phase7_residual = incremental_sub.add_parser(command)
        add_config_arg(phase7_residual)
        phase7_residual.add_argument("--latent-dim", type=int)
        phase7_residual.add_argument("--variant", default=None)
        phase7_residual.add_argument("--horizon-steps", type=int)
        phase7_residual.add_argument("--action-chunk-steps", type=int)
        phase7_residual.add_argument("--goal-encoding", choices=["absolute", "delta"], default=None)
        phase7_residual.add_argument("--goal-dropout-prob", type=float)
        phase7_residual.add_argument("--seed", type=int, default=0)
        phase7_residual.add_argument("--force", action="store_true")
        if command == "phase7-residual-replay-eval":
            phase7_residual.add_argument("--episodes", type=int)
        phase7_residual.set_defaults(func=incremental_cmd)
    phase7_flat = incremental_sub.add_parser("phase7-matched-flat-eval")
    add_config_arg(phase7_flat)
    phase7_flat.add_argument("--latent-dim", type=int)
    phase7_flat.add_argument("--variant", default=None)
    phase7_flat.add_argument("--seed", type=int, default=0)
    phase7_flat.add_argument("--episodes", type=int)
    phase7_flat.add_argument("--force", action="store_true")
    phase7_flat.set_defaults(func=incremental_cmd)
    phase7_goal_use = incremental_sub.add_parser("phase7-goal-use-eval")
    add_config_arg(phase7_goal_use)
    phase7_goal_use.add_argument("--latent-dim", type=int)
    phase7_goal_use.add_argument("--variant", default=None)
    phase7_goal_use.add_argument("--horizon-steps", type=int)
    phase7_goal_use.add_argument("--action-chunk-steps", type=int)
    phase7_goal_use.add_argument("--goal-encoding", choices=["absolute", "delta"], default=None)
    phase7_goal_use.add_argument("--goal-dropout-prob", type=float)
    phase7_goal_use.add_argument("--seed", type=int, default=0)
    phase7_goal_use.add_argument("--episodes", type=int)
    phase7_goal_use.add_argument("--dagger-iteration", type=int)
    phase7_goal_use.add_argument("--dagger-query-episodes", type=int)
    phase7_goal_use.add_argument("--counterfactual-queries", type=int, default=0)
    phase7_goal_use.add_argument("--force", action="store_true")
    phase7_goal_use.set_defaults(func=incremental_cmd)
    for command in ["phase7-priv-train", "phase7-priv-eval"]:
        phase7_priv = incremental_sub.add_parser(command)
        add_config_arg(phase7_priv)
        phase7_priv.add_argument("--horizon-steps", type=int)
        phase7_priv.add_argument("--seed", type=int, default=0)
        phase7_priv.add_argument("--force", action="store_true")
        if command == "phase7-priv-eval":
            phase7_priv.add_argument("--episodes", type=int)
        phase7_priv.set_defaults(func=incremental_cmd)
    for command in ["phase7-dagger-collect", "phase7-dagger-train", "phase7-dagger-eval"]:
        phase7_dagger = incremental_sub.add_parser(command)
        add_config_arg(phase7_dagger)
        phase7_dagger.add_argument("--latent-dim", type=int)
        phase7_dagger.add_argument("--variant", default=None)
        phase7_dagger.add_argument("--horizon-steps", type=int)
        phase7_dagger.add_argument("--action-chunk-steps", type=int)
        phase7_dagger.add_argument("--goal-encoding", choices=["absolute", "delta"], default=None)
        phase7_dagger.add_argument("--goal-dropout-prob", type=float)
        phase7_dagger.add_argument("--iteration", type=int, default=1)
        phase7_dagger.add_argument("--seed", type=int, default=0)
        if command in {"phase7-dagger-train", "phase7-dagger-eval"}:
            phase7_dagger.add_argument("--query-episodes", type=int)
        phase7_dagger.add_argument("--force", action="store_true")
        if command in {"phase7-dagger-collect", "phase7-dagger-eval"}:
            phase7_dagger.add_argument("--episodes", type=int)
        if command == "phase7-dagger-eval":
            phase7_dagger.add_argument(
                "--goal-mode",
                choices=["all", "correct", "shuffled", "zero"],
                default="all",
            )
        phase7_dagger.set_defaults(func=incremental_cmd)

    phase8_prepare = incremental_sub.add_parser("phase8-prepare")
    add_config_arg(phase8_prepare)
    phase8_prepare.add_argument("--latent-dim", type=int)
    phase8_prepare.add_argument("--variant", default=None)
    phase8_prepare.add_argument("--seed", type=int, default=0)
    phase8_prepare.add_argument("--force", action="store_true")
    phase8_prepare.set_defaults(func=incremental_cmd)
    phase8_train = incremental_sub.add_parser("phase8-train")
    add_config_arg(phase8_train)
    phase8_train.add_argument("--history", type=int, required=True)
    phase8_train.add_argument("--latent-dim", type=int)
    phase8_train.add_argument("--variant", default=None)
    phase8_train.add_argument("--horizon-steps", type=int)
    phase8_train.add_argument("--target-mode", choices=["absolute", "delta"], default="absolute")
    phase8_train.add_argument("--seed", type=int, default=0)
    phase8_train.add_argument("--force", action="store_true")
    phase8_train.set_defaults(func=incremental_cmd)
    phase8_structured = incremental_sub.add_parser("phase8-structured-train")
    add_config_arg(phase8_structured)
    phase8_structured.add_argument("--horizon-steps", type=int)
    phase8_structured.add_argument("--seed", type=int, default=0)
    phase8_structured.add_argument("--force", action="store_true")
    phase8_structured.set_defaults(func=incremental_cmd)
    phase8_structured_eval = incremental_sub.add_parser("phase8-structured-eval")
    add_config_arg(phase8_structured_eval)
    phase8_structured_eval.add_argument("--horizon-steps", type=int)
    phase8_structured_eval.add_argument("--seed", type=int, default=0)
    phase8_structured_eval.add_argument("--episodes", type=int)
    phase8_structured_eval.add_argument("--force", action="store_true")
    phase8_structured_eval.set_defaults(func=incremental_cmd)
    phase8_probe = incremental_sub.add_parser("phase8-probe-predictions")
    add_config_arg(phase8_probe)
    phase8_probe.add_argument("--latent-dim", type=int)
    phase8_probe.add_argument("--variant", default=None)
    phase8_probe.add_argument("--horizon-steps", type=int)
    phase8_probe.add_argument("--seed", type=int, default=0)
    phase8_probe.add_argument("--force", action="store_true")
    phase8_probe.set_defaults(func=incremental_cmd)
    phase8_dagger = incremental_sub.add_parser("phase8-dagger-train")
    add_config_arg(phase8_dagger)
    phase8_dagger.add_argument("--latent-dim", type=int)
    phase8_dagger.add_argument("--variant", default=None)
    phase8_dagger.add_argument("--horizon-steps", type=int)
    phase8_dagger.add_argument("--query-episodes", type=int, default=10)
    phase8_dagger.add_argument("--seed", type=int, default=0)
    phase8_dagger.add_argument("--force", action="store_true")
    phase8_dagger.set_defaults(func=incremental_cmd)
    phase8_dagger_collect = incremental_sub.add_parser("phase8-dagger-collect")
    add_config_arg(phase8_dagger_collect)
    phase8_dagger_collect.add_argument("--history", type=int, default=1)
    phase8_dagger_collect.add_argument("--latent-dim", type=int)
    phase8_dagger_collect.add_argument("--variant", default=None)
    phase8_dagger_collect.add_argument("--horizon-steps", type=int)
    phase8_dagger_collect.add_argument("--iteration", type=int, default=1)
    phase8_dagger_collect.add_argument("--episodes", type=int, default=10)
    phase8_dagger_collect.add_argument("--seed", type=int, default=0)
    phase8_dagger_collect.add_argument("--force", action="store_true")
    phase8_dagger_collect.set_defaults(func=incremental_cmd)
    phase8_low = incremental_sub.add_parser("phase8-low-adapt")
    add_config_arg(phase8_low)
    phase8_low.add_argument("--latent-dim", type=int)
    phase8_low.add_argument("--variant", default=None)
    phase8_low.add_argument("--horizon-steps", type=int)
    phase8_low.add_argument("--query-episodes", type=int, default=10)
    phase8_low.add_argument("--seed", type=int, default=0)
    phase8_low.add_argument("--force", action="store_true")
    phase8_low.set_defaults(func=incremental_cmd)
    phase8_action = incremental_sub.add_parser("phase8-action-train")
    add_config_arg(phase8_action)
    phase8_action.add_argument("--action-consistency-weight", type=float, required=True)
    phase8_action.add_argument("--latent-dim", type=int)
    phase8_action.add_argument("--variant", default=None)
    phase8_action.add_argument("--horizon-steps", type=int)
    phase8_action.add_argument("--seed", type=int, default=0)
    phase8_action.add_argument("--force", action="store_true")
    phase8_action.set_defaults(func=incremental_cmd)
    phase8_sweep = incremental_sub.add_parser("phase8-sweep")
    add_config_arg(phase8_sweep)
    phase8_sweep.add_argument("--latent-dim", type=int)
    phase8_sweep.add_argument("--variant", default=None)
    phase8_sweep.add_argument("--horizon-steps", type=int)
    phase8_sweep.add_argument("--histories", type=int, nargs="+")
    phase8_sweep.add_argument("--seed", type=int, default=0)
    phase8_sweep.add_argument("--force", action="store_true")
    phase8_sweep.set_defaults(func=incremental_cmd)
    phase8_eval = incremental_sub.add_parser("phase8-eval")
    add_config_arg(phase8_eval)
    phase8_eval.add_argument("--history", type=int, required=True)
    phase8_eval.add_argument("--latent-dim", type=int)
    phase8_eval.add_argument("--variant", default=None)
    phase8_eval.add_argument("--horizon-steps", type=int)
    phase8_eval.add_argument("--target-mode", choices=["absolute", "delta"], default="absolute")
    phase8_eval.add_argument("--seed", type=int, default=0)
    phase8_eval.add_argument("--episodes", type=int)
    phase8_eval.add_argument("--high-dagger-query-episodes", type=int)
    phase8_eval.add_argument("--adapted-low-query-episodes", type=int)
    phase8_eval.add_argument("--branch-action-weight", type=float, default=1.0)
    phase8_eval.add_argument("--action-consistency-weight", type=float)
    phase8_eval.add_argument("--force", action="store_true")
    phase8_eval.set_defaults(func=incremental_cmd)
    phase9_train = incremental_sub.add_parser("phase9-train")
    add_config_arg(phase9_train)
    phase9_train.add_argument("--latent-dim", type=int)
    phase9_train.add_argument("--variant", default=None)
    phase9_train.add_argument("--horizon-steps", type=int)
    phase9_train.add_argument("--trajectory-limit", type=int)
    phase9_train.add_argument("--seed", type=int, default=0)
    phase9_train.add_argument("--force", action="store_true")
    phase9_train.set_defaults(func=incremental_cmd)
    phase9_eval = incremental_sub.add_parser("phase9-eval")
    add_config_arg(phase9_eval)
    phase9_eval.add_argument("--sample-mode", choices=["zero", "random"], default="zero")
    phase9_eval.add_argument("--latent-dim", type=int)
    phase9_eval.add_argument("--variant", default=None)
    phase9_eval.add_argument("--horizon-steps", type=int)
    phase9_eval.add_argument("--seed", type=int, default=0)
    phase9_eval.add_argument("--episodes", type=int)
    phase9_eval.add_argument("--force", action="store_true")
    phase9_eval.set_defaults(func=incremental_cmd)
    phase10_collect = incremental_sub.add_parser("phase10-collect")
    add_config_arg(phase10_collect)
    phase10_collect.add_argument("--latent-dim", type=int)
    phase10_collect.add_argument("--variant", default=None)
    phase10_collect.add_argument("--horizon-steps", type=int)
    phase10_collect.add_argument("--seed", type=int, default=0)
    phase10_collect.add_argument("--episodes", type=int)
    phase10_collect.add_argument("--force", action="store_true")
    phase10_collect.set_defaults(func=incremental_cmd)
    for command in ["phase10-train", "phase10-eval"]:
        phase10 = incremental_sub.add_parser(command)
        add_config_arg(phase10)
        phase10.add_argument(
            "--method",
            choices=["direct", "interpolate", "empirical", "covariance_diag"],
            required=True,
        )
        phase10.add_argument("--latent-dim", type=int)
        phase10.add_argument("--variant", default=None)
        phase10.add_argument("--horizon-steps", type=int)
        phase10.add_argument("--interpolation-alpha", type=float, default=0.5)
        phase10.add_argument("--seed", type=int, default=0)
        phase10.add_argument("--query-episodes", type=int)
        if command == "phase10-eval":
            phase10.add_argument("--episodes", type=int)
        phase10.add_argument("--force", action="store_true")
        phase10.set_defaults(func=incremental_cmd)
    phase11 = incremental_sub.add_parser("phase11-eval")
    add_config_arg(phase11)
    phase11.add_argument("--seed", type=int, default=0)
    phase11.add_argument("--episodes", type=int, default=100)
    phase11.add_argument("--eval-seed-start", type=int, default=1_200_000)
    phase11.set_defaults(func=incremental_cmd)
    phase12_run = incremental_sub.add_parser("phase12-run")
    add_config_arg(phase12_run)
    phase12_run.add_argument("--n-trajectories", type=int, required=True)
    phase12_run.add_argument("--seed", type=int, default=0)
    phase12_run.add_argument("--episodes", type=int, default=100)
    phase12_run.add_argument("--eval-seed-start", type=int, default=1_200_000)
    phase12_run.set_defaults(func=incremental_cmd)
    phase12_plot = incremental_sub.add_parser("phase12-plot")
    add_config_arg(phase12_plot)
    phase12_plot.set_defaults(func=incremental_cmd)
    pre_rl_a_run = incremental_sub.add_parser("pre-rl-a-run")
    add_config_arg(pre_rl_a_run)
    pre_rl_a_run.add_argument("--seed", type=int, required=True)
    pre_rl_a_run.set_defaults(func=incremental_cmd)
    pre_rl_a_aggregate = incremental_sub.add_parser("pre-rl-a-aggregate")
    add_config_arg(pre_rl_a_aggregate)
    pre_rl_a_aggregate.set_defaults(func=incremental_cmd)
    pre_rl_b_train = incremental_sub.add_parser("pre-rl-b-train")
    add_config_arg(pre_rl_b_train)
    pre_rl_b_train.add_argument("--horizon-steps", type=int, required=True)
    pre_rl_b_train.add_argument("--seed", type=int, default=0)
    pre_rl_b_train.add_argument("--force", action="store_true")
    pre_rl_b_train.set_defaults(func=incremental_cmd)
    pre_rl_b_eval = incremental_sub.add_parser("pre-rl-b-eval")
    add_config_arg(pre_rl_b_eval)
    pre_rl_b_eval.add_argument("--horizon-steps", type=int, required=True)
    pre_rl_b_eval.add_argument("--seed", type=int, default=0)
    pre_rl_b_eval.add_argument("--episodes", type=int)
    pre_rl_b_eval.add_argument("--force", action="store_true")
    pre_rl_b_eval.set_defaults(func=incremental_cmd)
    pre_rl_b_aggregate = incremental_sub.add_parser("pre-rl-b-aggregate")
    add_config_arg(pre_rl_b_aggregate)
    pre_rl_b_aggregate.add_argument("--episodes", type=int)
    pre_rl_b_aggregate.set_defaults(func=incremental_cmd)
    pre_rl_c_oracle_sweep = incremental_sub.add_parser("pre-rl-c-oracle-sweep")
    add_config_arg(pre_rl_c_oracle_sweep)
    pre_rl_c_oracle_sweep.add_argument("--episodes", type=int)
    pre_rl_c_oracle_sweep.add_argument("--time-conditioned", action="store_true")
    pre_rl_c_oracle_sweep.add_argument("--horizons", type=int, nargs="+")
    pre_rl_c_oracle_sweep.add_argument("--force", action="store_true")
    pre_rl_c_oracle_sweep.set_defaults(func=incremental_cmd)
    pre_rl_c_train = incremental_sub.add_parser("pre-rl-c-train-time-conditioned")
    add_config_arg(pre_rl_c_train)
    pre_rl_c_train.add_argument("--horizon-steps", type=int, required=True)
    pre_rl_c_train.add_argument("--seed", type=int, default=0)
    pre_rl_c_train.add_argument("--force", action="store_true")
    pre_rl_c_train.set_defaults(func=incremental_cmd)
    pre_rl_d_collect = incremental_sub.add_parser("pre-rl-d-collect")
    add_config_arg(pre_rl_d_collect)
    pre_rl_d_collect.add_argument("--episodes", type=int)
    pre_rl_d_collect.add_argument("--force", action="store_true")
    pre_rl_d_collect.set_defaults(func=incremental_cmd)
    pre_rl_d_prepare = incremental_sub.add_parser("pre-rl-d-prepare")
    add_config_arg(pre_rl_d_prepare)
    pre_rl_d_prepare.add_argument("--episodes", type=int)
    pre_rl_d_prepare.add_argument("--force", action="store_true")
    pre_rl_d_prepare.set_defaults(func=incremental_cmd)
    pre_rl_d_manifests = incremental_sub.add_parser("pre-rl-d-manifests")
    add_config_arg(pre_rl_d_manifests)
    pre_rl_d_manifests.add_argument("--force", action="store_true")
    pre_rl_d_manifests.set_defaults(func=incremental_cmd)
    pre_rl_d_visual_bc = incremental_sub.add_parser("pre-rl-d-train-visual-bc")
    add_config_arg(pre_rl_d_visual_bc)
    pre_rl_d_visual_bc.add_argument(
        "--variant",
        required=True,
        choices=["clean", "mixed_25", "mixed_50", "recovery_heavy"],
    )
    pre_rl_d_visual_bc.add_argument("--label-view", choices=["query", "behavior"], default="query")
    pre_rl_d_visual_bc.add_argument(
        "--matched-hierarchy-data",
        action="store_true",
        help="Use the exact 60k Phase D hierarchy manifest for a matched flat comparison.",
    )
    pre_rl_d_visual_bc.add_argument("--seed", type=int, default=0)
    pre_rl_d_visual_bc.add_argument("--force", action="store_true")
    pre_rl_d_visual_bc.set_defaults(func=incremental_cmd)
    pre_rl_d_eval_visual_bc = incremental_sub.add_parser("pre-rl-d-eval-visual-bc")
    add_config_arg(pre_rl_d_eval_visual_bc)
    pre_rl_d_eval_visual_bc.add_argument(
        "--variant",
        required=True,
        choices=["clean", "mixed_25", "mixed_50", "recovery_heavy"],
    )
    pre_rl_d_eval_visual_bc.add_argument(
        "--label-view", choices=["query", "behavior"], default="query"
    )
    pre_rl_d_eval_visual_bc.add_argument(
        "--matched-hierarchy-data",
        action="store_true",
        help="Use the exact 60k Phase D hierarchy manifest for a matched flat comparison.",
    )
    pre_rl_d_eval_visual_bc.add_argument("--seed", type=int, default=0)
    pre_rl_d_eval_visual_bc.add_argument("--episodes", type=int)
    pre_rl_d_eval_visual_bc.add_argument("--force", action="store_true")
    pre_rl_d_eval_visual_bc.set_defaults(func=incremental_cmd)
    pre_rl_e_geometry = incremental_sub.add_parser("pre-rl-e-geometry")
    add_config_arg(pre_rl_e_geometry)
    pre_rl_e_geometry.set_defaults(func=incremental_cmd)
    pre_rl_f_train = incremental_sub.add_parser("pre-rl-f-train-privileged-tcp")
    add_config_arg(pre_rl_f_train)
    pre_rl_f_train.add_argument("--seed", type=int, default=0)
    pre_rl_f_train.add_argument("--force", action="store_true")
    pre_rl_f_train.set_defaults(func=incremental_cmd)
    pre_rl_f_eval = incremental_sub.add_parser("pre-rl-f-eval-privileged-tcp")
    add_config_arg(pre_rl_f_eval)
    pre_rl_f_eval.add_argument("--seed", type=int, default=0)
    pre_rl_f_eval.add_argument("--episodes", type=int)
    pre_rl_f_eval.add_argument("--force", action="store_true")
    pre_rl_f_eval.set_defaults(func=incremental_cmd)
    pre_rl_f_visual_train = incremental_sub.add_parser("pre-rl-f-train-visual-tcp")
    add_config_arg(pre_rl_f_visual_train)
    pre_rl_f_visual_train.add_argument("--representation", choices=["raw", "ae256"], required=True)
    pre_rl_f_visual_train.add_argument("--seed", type=int, default=0)
    pre_rl_f_visual_train.add_argument("--force", action="store_true")
    pre_rl_f_visual_train.set_defaults(func=incremental_cmd)
    pre_rl_f_visual_eval = incremental_sub.add_parser("pre-rl-f-eval-visual-tcp")
    add_config_arg(pre_rl_f_visual_eval)
    pre_rl_f_visual_eval.add_argument("--representation", choices=["raw", "ae256"], required=True)
    pre_rl_f_visual_eval.add_argument(
        "--goal-source", choices=["learned", "oracle"], default="learned"
    )
    pre_rl_f_visual_eval.add_argument("--audit-branch", action="store_true")
    pre_rl_f_visual_eval.add_argument("--seed", type=int, default=0)
    pre_rl_f_visual_eval.add_argument("--episodes", type=int)
    pre_rl_f_visual_eval.add_argument("--force", action="store_true")
    pre_rl_f_visual_eval.set_defaults(func=incremental_cmd)
    pre_rl_f_visual_record = incremental_sub.add_parser("pre-rl-f-record-visual-tcp")
    add_config_arg(pre_rl_f_visual_record)
    pre_rl_f_visual_record.add_argument("--representation", choices=["raw", "ae256"], required=True)
    pre_rl_f_visual_record.add_argument(
        "--goal-source", choices=["learned", "oracle"], default="learned"
    )
    pre_rl_f_visual_record.add_argument("--seed", type=int, default=0)
    pre_rl_f_visual_record.add_argument("--episodes", type=int, default=10)
    pre_rl_f_visual_record.add_argument("--eval-seed-start", type=int)
    pre_rl_f_visual_record.add_argument("--force", action="store_true")
    pre_rl_f_visual_record.set_defaults(func=incremental_cmd)
    pre_rl_d_hierarchy_manifests = incremental_sub.add_parser("pre-rl-d-hierarchy-manifests")
    add_config_arg(pre_rl_d_hierarchy_manifests)
    pre_rl_d_hierarchy_manifests.add_argument("--force", action="store_true")
    pre_rl_d_hierarchy_manifests.set_defaults(func=incremental_cmd)
    pre_rl_d_train_hierarchy = incremental_sub.add_parser("pre-rl-d-train-hierarchy")
    add_config_arg(pre_rl_d_train_hierarchy)
    pre_rl_d_train_hierarchy.add_argument("--variant", choices=["clean", "mixed_25"], required=True)
    pre_rl_d_train_hierarchy.add_argument("--seed", type=int, default=0)
    pre_rl_d_train_hierarchy.add_argument("--force", action="store_true")
    pre_rl_d_train_hierarchy.set_defaults(func=incremental_cmd)
    pre_rl_d_eval_hierarchy = incremental_sub.add_parser("pre-rl-d-eval-hierarchy")
    add_config_arg(pre_rl_d_eval_hierarchy)
    pre_rl_d_eval_hierarchy.add_argument("--variant", choices=["clean", "mixed_25"], required=True)
    pre_rl_d_eval_hierarchy.add_argument("--disturbed", action="store_true")
    pre_rl_d_eval_hierarchy.add_argument("--seed", type=int, default=0)
    pre_rl_d_eval_hierarchy.add_argument("--episodes", type=int)
    pre_rl_d_eval_hierarchy.add_argument("--force", action="store_true")
    pre_rl_d_eval_hierarchy.set_defaults(func=incremental_cmd)
    pre_rl_g_tcp_diagnostics = incremental_sub.add_parser("pre-rl-g-tcp-diagnostics")
    add_config_arg(pre_rl_g_tcp_diagnostics)
    pre_rl_g_tcp_diagnostics.add_argument("--force", action="store_true")
    pre_rl_g_tcp_diagnostics.set_defaults(func=incremental_cmd)
    for command in [
        "learned-interface-train-representation",
        "learned-interface-probe",
        "learned-interface-prepare",
        "learned-interface-train-hierarchy",
        "learned-interface-eval",
        "learned-interface-run",
        "learned-interface-record",
    ]:
        learned_interface = incremental_sub.add_parser(command)
        add_config_arg(learned_interface)
        learned_interface.add_argument("--candidate", required=True)
        learned_interface.add_argument("--seed", type=int, default=0)
        learned_interface.add_argument("--force", action="store_true")
        if command in {
            "learned-interface-eval",
            "learned-interface-run",
            "learned-interface-record",
        }:
            learned_interface.add_argument("--episodes", type=int)
        if command in {"learned-interface-eval", "learned-interface-record"}:
            learned_interface.add_argument(
                "--goal-source",
                choices=["learned", "oracle"],
                required=True,
            )
        if command == "learned-interface-record":
            learned_interface.add_argument("--eval-seed-start", type=int)
        learned_interface.set_defaults(func=incremental_cmd)

    vae_scaling_manifests = incremental_sub.add_parser("vae-scaling-manifests")
    add_config_arg(vae_scaling_manifests)
    vae_scaling_manifests.set_defaults(func=incremental_cmd)
    vae_scaling_extend = incremental_sub.add_parser("vae-scaling-extend-data")
    add_config_arg(vae_scaling_extend)
    vae_scaling_extend.add_argument("--force", action="store_true")
    vae_scaling_extend.set_defaults(func=incremental_cmd)
    vae_scaling_aggregate = incremental_sub.add_parser("vae-scaling-aggregate")
    add_config_arg(vae_scaling_aggregate)
    vae_scaling_aggregate.add_argument("--episodes", type=int, default=500)
    vae_scaling_aggregate.add_argument("--oracle-episodes", type=int, default=50)
    vae_scaling_aggregate.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    vae_scaling_aggregate.add_argument("--output-name", default="aggregate")
    vae_scaling_aggregate.set_defaults(func=incremental_cmd)
    for command in [
        "vae-scaling-train",
        "vae-scaling-eval",
        "vae-scaling-run",
    ]:
        vae_scaling = incremental_sub.add_parser(command)
        add_config_arg(vae_scaling)
        vae_scaling.add_argument("--n-trajectories", type=int, required=True)
        vae_scaling.add_argument("--seed", type=int, required=True)
        vae_scaling.add_argument("--force", action="store_true")
        if command != "vae-scaling-train":
            vae_scaling.add_argument("--episodes", type=int)
            vae_scaling.add_argument("--oracle-episodes", type=int)
        vae_scaling.set_defaults(func=incremental_cmd)

    p = sub.add_parser("train")
    add_config_arg(p)
    p.add_argument(
        "kind",
        choices=[
            "encoder",
            "flat",
            "flat_obs",
            "bc_obs",
            "bc_obs_1step",
            "bc_obs_dagger",
            "bc_pose",
            "bc_state",
            "high",
            "low",
        ],
    )
    p.add_argument("--n-traj", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-s", type=float)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=train_cmd)

    p = sub.add_parser("eval")
    add_config_arg(p)
    p.add_argument(
        "method",
        choices=[
            "flat",
            "flat_obs",
            "bc_obs",
            "bc_obs_1step",
            "bc_obs_dagger",
            "bc_pose",
            "bc_state",
            "hier",
        ],
    )
    p.add_argument("--n-traj", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-s", type=float)
    p.add_argument("--episodes", type=int)
    p.set_defaults(func=eval_cmd)

    p = sub.add_parser("video")
    add_config_arg(p)
    p.add_argument("method", choices=["flat", "flat_obs", "hier"])
    p.add_argument("--n-traj", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--horizon-s", type=float)
    p.set_defaults(func=video_cmd)

    p = sub.add_parser("run-sweep")
    add_config_arg(p)
    p.add_argument("--profile", choices=["staged", "full"], default="staged")
    p.set_defaults(func=run_sweep)

    p = sub.add_parser("report")
    add_config_arg(p)
    p.set_defaults(func=report_cmd)

    p = sub.add_parser("probe-latent")
    add_config_arg(p)
    p.add_argument("--n-traj", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.set_defaults(func=probe_cmd)

    p = sub.add_parser("diagnose-hier")
    add_config_arg(p)
    p.add_argument("--n-traj", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-s", type=float, required=True)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--out", required=True)
    p.set_defaults(func=diagnose_cmd)

    p = sub.add_parser("commit")
    p.add_argument("-m", "--message", required=True)
    p.set_defaults(func=commit_cmd)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
