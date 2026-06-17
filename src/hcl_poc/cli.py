from __future__ import annotations

import argparse
import subprocess
import sys

import torch
from rich.console import Console

from hcl_poc.config import load_config
from hcl_poc.data import prepare_dataset
from hcl_poc.eval import evaluate, horizon_steps, record_videos
from hcl_poc.report import build_report
from hcl_poc.rl import collect_ppo_dataset, evaluate_ppo, ppo_status, train_ppo
from hcl_poc.train import (
    train_bc_policy,
    train_dagger_bc_policy,
    train_flow_policy,
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

        env = gym.make(config.get("env_id"), obs_mode=config.get("obs_mode"), control_mode=config.get("control_mode"))
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
    evaluate(config, args.n_traj, args.seed, args.method, args.horizon_s)


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

    p = sub.add_parser("train")
    add_config_arg(p)
    p.add_argument(
        "kind",
        choices=["encoder", "flat", "flat_obs", "bc_obs", "bc_obs_1step", "bc_obs_dagger", "bc_state", "high", "low"],
    )
    p.add_argument("--n-traj", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-s", type=float)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=train_cmd)

    p = sub.add_parser("eval")
    add_config_arg(p)
    p.add_argument("method", choices=["flat", "flat_obs", "bc_obs", "bc_obs_1step", "bc_obs_dagger", "bc_state", "hier"])
    p.add_argument("--n-traj", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-s", type=float)
    p.set_defaults(func=eval_cmd)

    p = sub.add_parser("video")
    add_config_arg(p)
    p.add_argument("method", choices=["flat_obs"])
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
