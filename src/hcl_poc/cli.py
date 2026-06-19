from __future__ import annotations

import argparse
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
    probe_phase6_representation,
    probe_phase4_visual_history,
    prepare_phase8_latent_episodes,
    run_phase0,
    run_phase7_branch_audit,
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
    sweep_phase8_deterministic_predictors,
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

console = Console()


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pusht.yaml")


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
            seed=args.seed,
            episodes=args.episodes,
            high_dagger_query_episodes=args.high_dagger_query_episodes,
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
    phase8_train.add_argument("--seed", type=int, default=0)
    phase8_train.add_argument("--force", action="store_true")
    phase8_train.set_defaults(func=incremental_cmd)
    phase8_dagger = incremental_sub.add_parser("phase8-dagger-train")
    add_config_arg(phase8_dagger)
    phase8_dagger.add_argument("--latent-dim", type=int)
    phase8_dagger.add_argument("--variant", default=None)
    phase8_dagger.add_argument("--horizon-steps", type=int)
    phase8_dagger.add_argument("--query-episodes", type=int, default=10)
    phase8_dagger.add_argument("--seed", type=int, default=0)
    phase8_dagger.add_argument("--force", action="store_true")
    phase8_dagger.set_defaults(func=incremental_cmd)
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
    phase8_eval.add_argument("--seed", type=int, default=0)
    phase8_eval.add_argument("--episodes", type=int)
    phase8_eval.add_argument("--high-dagger-query-episodes", type=int)
    phase8_eval.add_argument("--force", action="store_true")
    phase8_eval.set_defaults(func=incremental_cmd)

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
