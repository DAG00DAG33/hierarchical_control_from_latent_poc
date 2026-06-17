from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill  # noqa: F401
import numpy as np
import torch
from rich.console import Console
from torch import nn
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.features import DinoExtractor, batched
from hcl_poc.utils import Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).flatten()
        return action, logprob, entropy, value


@dataclass(frozen=True)
class PPOPaths:
    latest: Path
    best: Path
    metrics: Path


def _rl_paths(config: Config) -> PPOPaths:
    out_dir = ensure_dir(config.path_value("paths.rl_dir"))
    return PPOPaths(
        latest=out_dir / "ppo_latest.pt",
        best=out_dir / "ppo_best.pt",
        metrics=out_dir / "ppo_metrics.json",
    )


def _rl_backend(config: Config) -> str:
    backend = str(config.get("rl.sim_backend", "auto"))
    if backend == "auto":
        return "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    return backend


def _make_state_env(config: Config, num_envs: int, record_metrics: bool = True):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        config.get("env_id"),
        obs_mode="state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        sim_backend=_rl_backend(config),
        num_envs=num_envs,
        reconfiguration_freq=int(config.get("rl.reconfiguration_freq", 1)),
    )
    return ManiSkillVectorEnv(
        base,
        num_envs,
        ignore_terminations=True,
        record_metrics=record_metrics,
    )


def _scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().numpy().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_agent(
    path: Path,
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    metrics: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "hidden_dim": agent.actor_mean[0].out_features,
            "global_step": global_step,
            "metrics": metrics,
        },
        path,
    )


def load_ppo_agent(path: str | Path, device: torch.device | None = None) -> PPOAgent:
    device = device or default_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent = PPOAgent(int(ckpt["obs_dim"]), int(ckpt["action_dim"]), int(ckpt["hidden_dim"])).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent


def train_ppo(config: Config, resume: bool = True) -> Path:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = default_device()
    if not torch.cuda.is_available() and _rl_backend(config) == "physx_cuda":
        raise RuntimeError("rl.sim_backend resolved to physx_cuda but CUDA is unavailable")

    num_envs = int(config.get("rl.num_envs", 1024))
    num_steps = int(config.get("rl.num_steps", 100))
    total_timesteps = int(config.get("rl.total_timesteps", 5_000_000))
    if _rl_backend(config) == "physx_cpu" and not bool(config.get("rl.allow_cpu", False)):
        raise RuntimeError(
            "CUDA is unavailable, so rl.sim_backend resolved to physx_cpu. "
            "Full PPO training is intentionally disabled on CPU; restore the NVIDIA driver "
            "or set rl.allow_cpu: true for a small smoke/debug run."
        )
    batch_size = num_envs * num_steps
    minibatches = int(config.get("rl.num_minibatches", 32))
    minibatch_size = batch_size // minibatches
    if batch_size % minibatches != 0:
        raise ValueError("rl.num_envs * rl.num_steps must divide rl.num_minibatches")

    env = _make_state_env(config, num_envs)
    obs_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = PPOAgent(obs_dim, action_dim, int(config.get("rl.hidden_dim", 256))).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=float(config.get("rl.learning_rate", 3e-4)),
        eps=1e-5,
    )
    paths = _rl_paths(config)
    global_step = 0
    best_success = -1.0
    if resume and paths.latest.exists():
        ckpt = torch.load(paths.latest, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", 0))
        best_success = float(ckpt.get("metrics", {}).get("best_success", -1.0))
        console.print(f"Resuming PPO from {paths.latest} at step {global_step}")

    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs, action_dim), device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs, _info = env.reset(seed=seed)
    next_obs = next_obs.to(device).float()
    next_done = torch.zeros(num_envs, device=device)
    num_updates = max(1, (total_timesteps - global_step) // batch_size)
    gamma = float(config.get("rl.gamma", 0.8))
    gae_lambda = float(config.get("rl.gae_lambda", 0.9))
    update_epochs = int(config.get("rl.update_epochs", 8))
    clip_coef = float(config.get("rl.clip_coef", 0.2))
    ent_coef = float(config.get("rl.ent_coef", 0.0))
    vf_coef = float(config.get("rl.vf_coef", 0.5))
    max_grad_norm = float(config.get("rl.max_grad_norm", 0.5))
    target_kl = config.get("rl.target_kl")
    target_kl_f = None if target_kl is None else float(target_kl)
    anneal_lr = bool(config.get("rl.anneal_lr", True))
    timer = Timer()
    recent_successes: list[float] = []
    latest_metrics: dict[str, Any] = {}

    for update in trange(1, num_updates + 1, desc="train privileged PPO"):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * float(config.get("rl.learning_rate", 3e-4))

        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _entropy, value = agent.get_action_and_value(next_obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.to(device).float()
            rewards_buf[step] = reward.to(device).view(-1)
            next_done = torch.logical_or(terminated, truncated).to(device).float().view(-1)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    successes = info["final_info"]["episode"]["success_once"][mask].detach().float().cpu().numpy()
                    recent_successes.extend(float(x) for x in successes)
                    recent_successes = recent_successes[-200:]

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)
        for _epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]
                _new_action, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
            if target_kl_f is not None and approx_kl > target_kl_f:
                break

        success = float(np.mean(recent_successes)) if recent_successes else 0.0
        latest_metrics = {
            "global_step": global_step,
            "success_recent": success,
            "best_success": max(best_success, success),
            "mean_reward": float(rewards_buf.mean().detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "old_approx_kl": float(old_approx_kl.detach().cpu()),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "elapsed_s": timer.elapsed(),
            "backend": _rl_backend(config),
            "num_envs": num_envs,
        }
        if update % int(config.get("rl.save_every_updates", 10)) == 0 or update == num_updates:
            _save_agent(paths.latest, agent, optimizer, global_step, latest_metrics)
            write_json(paths.metrics, latest_metrics)
        if success > best_success:
            best_success = success
            latest_metrics["best_success"] = best_success
            _save_agent(paths.best, agent, optimizer, global_step, latest_metrics)
            write_json(paths.metrics, latest_metrics)
        if success >= float(config.get("rl.target_success", 0.9)):
            console.print(f"Reached PPO target success {success:.3f} at step {global_step}")
            break

    _save_agent(paths.latest, agent, optimizer, global_step, latest_metrics)
    if not paths.best.exists():
        _save_agent(paths.best, agent, optimizer, global_step, latest_metrics)
    write_json(paths.metrics, latest_metrics)
    env.close()
    return paths.best


@torch.inference_mode()
def evaluate_ppo(config: Config, checkpoint: str | Path | None = None, episodes: int | None = None) -> dict[str, float]:
    device = default_device()
    path = Path(checkpoint) if checkpoint is not None else _rl_paths(config).best
    agent = load_ppo_agent(path, device)
    num_envs = int(config.get("rl.eval_num_envs", config.get("rl.num_envs", 64)))
    episodes = int(episodes or config.get("rl.eval_episodes", 256))
    env = _make_state_env(config, num_envs)
    obs, _info = env.reset(seed=int(config.get("rl.eval_seed", 50_000)))
    obs = obs.to(device).float()
    successes: list[float] = []
    returns: list[float] = []
    while len(successes) < episodes:
        action, _logprob, _entropy, _value = agent.get_action_and_value(obs, deterministic=True)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        obs = obs.to(device).float()
        if "final_info" in info:
            mask = info["_final_info"]
            if bool(mask.any()):
                ep = info["final_info"]["episode"]
                successes.extend(float(x) for x in ep["success_once"][mask].detach().float().cpu().numpy())
                returns.extend(float(x) for x in ep["return"][mask].detach().float().cpu().numpy())
    env.close()
    metrics = {
        "success": float(np.mean(successes[:episodes])),
        "return": float(np.mean(returns[:episodes])),
        "episodes": float(episodes),
    }
    console.print(metrics)
    return metrics


def _rgb_and_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = _to_numpy(obs["sensor_data"]["base_camera"]["rgb"])
    state = _to_numpy(obs["state"])
    if rgb.ndim == 4:
        rgb = rgb[0]
    if state.ndim == 2:
        state = state[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8), state.astype(np.float32)


@torch.inference_mode()
def collect_ppo_dataset(
    config: Config,
    checkpoint: str | Path | None = None,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    out_path = config.path_value("paths.prepared_path")
    required = int(episodes or config.get("data.max_trajectories", 200))
    if out_path.exists() and not force:
        with h5py.File(out_path, "r") as h5:
            existing = len([k for k in h5 if k.startswith("episode_")])
        if existing >= required:
            console.print(f"Prepared policy dataset already exists: {out_path}")
            return out_path

    device = default_device()
    path = Path(checkpoint) if checkpoint is not None else _rl_paths(config).best
    min_success = config.get("rl.collect_min_success")
    if min_success is not None:
        eval_metrics = evaluate_ppo(
            config,
            checkpoint=path,
            episodes=int(config.get("rl.collect_eval_episodes", config.get("rl.eval_episodes", 256))),
        )
        if eval_metrics["success"] < float(min_success):
            raise RuntimeError(
                f"PPO checkpoint {path} has success={eval_metrics['success']:.3f}, "
                f"below rl.collect_min_success={float(min_success):.3f}; not collecting demos."
            )
    agent = load_ppo_agent(path, device)
    extractor = DinoExtractor(config.get("dino.model_name"), device)
    batch_size = int(config.get("dino.batch_size", 32))
    max_attempts = int(config.get("rl.collect_max_attempts", required * 20))
    env = gym.make(
        config.get("env_id"),
        obs_mode="rgb+state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=_rl_backend(config),
        num_envs=1,
        reconfiguration_freq=int(config.get("rl.reconfiguration_freq", 1)),
    )
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    successes = 0
    attempts = 0
    with h5py.File(tmp_path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["source"] = "privileged_ppo"
        meta.attrs["checkpoint"] = str(path)
        meta.attrs["dino_model"] = config.get("dino.model_name")
        meta.attrs["control_mode"] = config.get("control_mode")
        meta.attrs["obs_mode"] = "rgb+state"
        for attempts in trange(1, max_attempts + 1, desc="collect PPO demos"):
            obs, _info = env.reset(seed=int(config.get("rl.collect_seed", 70_000)) + attempts)
            rgbs: list[np.ndarray] = []
            proprios: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            success = False
            done = False
            truncated = False
            while not (done or truncated):
                rgb, state = _rgb_and_state(obs)
                state_t = torch.from_numpy(state[None]).to(device).float()
                action_t, _logprob, _entropy, _value = agent.get_action_and_value(
                    state_t,
                    deterministic=True,
                )
                action = action_t.detach().cpu().numpy()[0].astype(np.float32)
                rgbs.append(rgb)
                proprios.append(state[:21].copy())
                actions.append(action)
                obs, _reward, done, truncated, info = env.step(action)
                success = success or _scalar_bool(info.get("success", False))
            if not success:
                continue
            rgb_arr = np.stack(rgbs, axis=0)
            feats = [extractor.encode_batch(chunk) for chunk in batched(rgb_arr, batch_size)]
            ep = h5.create_group(f"episode_{successes:04d}")
            ep.create_dataset("dino", data=np.concatenate(feats, axis=0), compression="gzip")
            ep.create_dataset("proprio", data=np.stack(proprios, axis=0), compression="gzip")
            ep.create_dataset("actions", data=np.stack(actions, axis=0), compression="gzip")
            successes += 1
            if successes >= required:
                break
        meta.attrs["attempts"] = attempts
        meta.attrs["successes"] = successes
    env.close()
    if successes < required:
        raise RuntimeError(f"Collected only {successes}/{required} successful PPO demos")
    tmp_path.replace(out_path)
    console.print(f"Wrote PPO-collected prepared dataset: {out_path}")
    return out_path
