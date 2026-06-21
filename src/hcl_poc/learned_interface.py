from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.incremental import (
    _load_phase6_train_episodes,
    _binary_auc,
    _phase4_dino_from_config,
    _phase4_fit_standardizers,
    _phase4_frame_inputs,
    _phase6_train_probe_heads,
    _phase7_obs_state_tensor,
    _rl_backend,
    _runtime_metadata,
    _wilson_interval,
    collect_phase6_probe_dataset,
)
from hcl_poc.models import (
    MLP,
    ObservationEncoder,
    RepresentationWorldModel,
    VariationalObservationEncoder,
)
from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def _candidate_specs(config: Config) -> dict[str, dict[str, Any]]:
    specs = config.get("learned_interface.candidates")
    if not isinstance(specs, dict):
        raise ValueError("learned_interface.candidates must be a mapping")
    return {str(name): dict(value) for name, value in specs.items()}


def learned_interface_candidate_spec(config: Config, candidate: str) -> dict[str, Any]:
    specs = _candidate_specs(config)
    if candidate not in specs:
        raise ValueError(
            f"Unknown learned-interface candidate {candidate!r}; "
            f"available: {sorted(specs)}"
        )
    spec = specs[candidate]
    spec.setdefault("family", "vae")
    spec.setdefault("encoder_type", "vae")
    spec.setdefault("latent_dim", 256)
    spec.setdefault("width", 1024)
    spec.setdefault("beta", 0.0)
    spec.setdefault("kl_warmup_steps", 0)
    spec.setdefault("free_bits", 0.0)
    spec.setdefault("dino_noise_std", 0.0)
    spec.setdefault("proprio_noise_std", 0.0)
    spec.setdefault("reconstruction_weight", 0.1)
    spec.setdefault("representation_candidate", candidate)
    spec.setdefault("high_level_candidate", candidate)
    spec.setdefault("conditioning", "concat")
    spec.setdefault("predictor_width", spec["width"])
    spec.setdefault("ema_momentum", 0.99)
    spec.setdefault("lambda_pred", 1.0)
    spec.setdefault("lambda_var", 1.0)
    spec.setdefault("lambda_cov", 0.01)
    spec.setdefault("lambda_recon", 0.1)
    spec.setdefault("horizon_offsets", [1, 2, 5, 10])
    spec.setdefault("early_stopping_patience", 10)
    spec.setdefault("lambda_action", 1.0)
    spec.setdefault("lambda_auxiliary", 1.0)
    spec["horizon_offsets"] = [
        int(value) for value in spec["horizon_offsets"]
    ]
    return spec


def _artifact_dir(config: Config, candidate: str, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "learned_interface"
        / candidate
        / f"seed{seed}"
    )


def _result_dir(config: Config, candidate: str, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "learned_interface"
        / candidate
        / f"seed{seed}"
    )


def _write_representation_metrics(path: Path, payload: dict[str, Any]) -> None:
    write_json(
        path,
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "encoder",
                "target_encoder",
                "world_model",
                "decoder",
                "frame_norm",
                "action_norm",
            }
        },
    )


class _RepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, episodes: list[dict[str, np.ndarray]], length: int) -> None:
        self.episodes = [episode for episode in episodes if len(episode["actions"]) > 0]
        self.length = length
        if not self.episodes:
            raise ValueError("No learned-interface representation episodes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> torch.Tensor:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        t = int(np.random.randint(0, len(episode["frames"])))
        return torch.from_numpy(episode["frames"][t])


class _PredictiveRepresentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        horizons: list[int],
        length: int,
    ) -> None:
        self.episodes = [
            episode
            for episode in episodes
            if len(episode["actions"]) > min(horizons)
        ]
        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.length = length
        if not self.episodes:
            raise ValueError("No predictive learned-interface episodes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        valid_horizons = [
            horizon
            for horizon in self.horizons
            if len(episode["actions"]) > horizon
        ]
        horizon = int(
            valid_horizons[np.random.randint(0, len(valid_horizons))]
        )
        t = int(np.random.randint(0, len(episode["actions"]) - horizon))
        actions = np.zeros(
            (self.max_horizon, episode["actions"].shape[-1]),
            dtype=np.float32,
        )
        actions[:horizon] = episode["actions"][t : t + horizon]
        return {
            "x_t": torch.from_numpy(episode["frames"][t]),
            "x_future": torch.from_numpy(episode["frames"][t + horizon]),
            "actions": torch.from_numpy(actions),
            "horizon": torch.tensor(horizon, dtype=torch.long),
        }


class _EffectEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        effect_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.model = MLP(
            2 * input_dim + 1,
            effect_dim,
            hidden_dim,
            depth=3,
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.model(pair)


class _EffectRepresentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        horizon_steps: int,
        length: int,
    ) -> None:
        self.episodes = [
            episode
            for episode in episodes
            if len(episode["actions"]) > horizon_steps
        ]
        self.horizon_steps = horizon_steps
        self.length = length
        if not self.episodes:
            raise ValueError("No action-aware effect-code episodes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        base = int(
            np.random.randint(
                0, len(episode["actions"]) - self.horizon_steps
            )
        )
        offset = int(np.random.randint(0, self.horizon_steps))
        current = base + offset
        previous = (
            episode["actions"][current - 1]
            if current > 0
            else episode["zero_action"]
        )
        return {
            "x_start": torch.from_numpy(episode["frames"][base]),
            "x_future": torch.from_numpy(
                episode["frames"][base + self.horizon_steps]
            ),
            "x_current": torch.from_numpy(episode["frames"][current]),
            "previous": torch.from_numpy(previous),
            "remaining": torch.tensor(
                (self.horizon_steps - offset) / self.horizon_steps,
                dtype=torch.float32,
            ),
            "action": torch.from_numpy(episode["actions"][current]),
            "auxiliary": torch.from_numpy(
                episode["auxiliary"][base + self.horizon_steps]
            ),
        }


def _reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    proprio_dim: int,
    proprio_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dino = torch.mean(
        (prediction[:, :-proprio_dim] - target[:, :-proprio_dim]) ** 2
    )
    proprio = torch.mean(
        (prediction[:, -proprio_dim:] - target[:, -proprio_dim:]) ** 2
    )
    return dino + proprio_weight * proprio, dino, proprio


def _vae_kl(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_dimension = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
    optimized = torch.clamp(per_dimension, min=free_bits)
    return optimized.sum(dim=-1).mean(), per_dimension.mean()


def _variance_covariance_losses(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    variance = torch.relu(1.0 - std).mean()
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / max(len(z) - 1, 1)
    covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = covariance.square().sum() / z.shape[-1]
    return variance, covariance_loss


def _train_predictive_representation(
    config: Config,
    candidate: str,
    spec: dict[str, Any],
    seed: int,
    force: bool,
) -> Path:
    artifact_dir = _artifact_dir(config, candidate, seed)
    checkpoint_path = artifact_dir / "representation.pt"
    if checkpoint_path.exists() and not force:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        _write_representation_metrics(
            artifact_dir / "representation_metrics.json", checkpoint
        )
        console.print(f"Learned-interface representation exists: {checkpoint_path}")
        return checkpoint_path
    set_seed(seed)
    train_episodes, validation_episodes, data_metadata = (
        _load_phase6_train_episodes(config)
    )
    frame_norm, action_norm = _phase4_fit_standardizers(train_episodes)

    def normalized(
        episodes: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        return [
            {
                "frames": frame_norm.transform(episode["frames"]),
                "actions": action_norm.transform(episode["actions"]),
            }
            for episode in episodes
        ]

    train = normalized(train_episodes)
    validation = normalized(validation_episodes)
    input_dim = int(data_metadata["frame_dim"])
    action_dim = int(data_metadata["action_dim"])
    latent_dim = int(spec["latent_dim"])
    width = int(spec["width"])
    predictor_width = int(spec["predictor_width"])
    horizons = [int(value) for value in spec["horizon_offsets"]]
    device = default_device()
    encoder = ObservationEncoder(input_dim, latent_dim, width).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.requires_grad_(False)
    predictor = RepresentationWorldModel(
        latent_dim, action_dim, predictor_width
    ).to(device)
    lambda_recon = float(spec["lambda_recon"])
    decoder = (
        MLP(latent_dim, input_dim, width, depth=3).to(device)
        if lambda_recon > 0.0
        else None
    )
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    if decoder is not None:
        parameters += list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learned_interface.representation.lr", 3e-4)),
    )
    batch_size = int(config.get("learned_interface.representation.batch_size", 512))
    batches_per_epoch = int(
        spec.get(
            "batches_per_epoch",
            config.get("learned_interface.representation.batches_per_epoch", 400),
        )
    )
    epochs = int(
        spec.get(
            "epochs",
            config.get("learned_interface.representation.epochs", 60),
        )
    )
    train_loader = DataLoader(
        _PredictiveRepresentationDataset(
            train, horizons, batch_size * batches_per_epoch
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        _PredictiveRepresentationDataset(
            validation,
            horizons,
            int(
                config.get(
                    "learned_interface.representation.validation_samples",
                    8192,
                )
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    proprio_dim = int(config.get("incremental.phase6.proprio_dim", 21))
    proprio_weight = float(
        config.get("incremental.phase6.proprio_reconstruction_weight", 1.0)
    )
    lambda_pred = float(spec["lambda_pred"])
    lambda_var = float(spec["lambda_var"])
    lambda_cov = float(spec["lambda_cov"])
    ema_momentum = float(spec["ema_momentum"])
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = int(spec["early_stopping_patience"])
    timer = Timer()

    @torch.no_grad()
    def update_target() -> None:
        for target_parameter, parameter in zip(
            target_encoder.parameters(), encoder.parameters(), strict=True
        ):
            target_parameter.lerp_(parameter, 1.0 - ema_momentum)

    def validation_metrics() -> dict[str, Any]:
        prediction_rows = []
        reconstruction_rows = []
        dino_rows = []
        proprio_rows = []
        variance_rows = []
        covariance_rows = []
        latent_rows = []
        with torch.inference_mode():
            for batch in validation_loader:
                x_t = batch["x_t"].to(device).float()
                x_future = batch["x_future"].to(device).float()
                actions = batch["actions"].to(device).float()
                horizon = batch["horizon"].to(device)
                z_t = encoder(x_t)
                z_future = encoder(x_future)
                target = target_encoder(x_future)
                prediction = predictor(z_t, actions, horizon)
                prediction_rows.append(
                    float(torch.mean((prediction - target) ** 2).cpu())
                )
                variance_t, covariance_t = _variance_covariance_losses(z_t)
                variance_future, covariance_future = (
                    _variance_covariance_losses(z_future)
                )
                variance_rows.append(
                    float((0.5 * (variance_t + variance_future)).cpu())
                )
                covariance_rows.append(
                    float((0.5 * (covariance_t + covariance_future)).cpu())
                )
                if decoder is not None:
                    reconstruction_t = decoder(z_t)
                    reconstruction_future = decoder(z_future)
                    total_t, dino_t, proprio_t = _reconstruction_loss(
                        reconstruction_t, x_t, proprio_dim, proprio_weight
                    )
                    total_future, dino_future, proprio_future = (
                        _reconstruction_loss(
                            reconstruction_future,
                            x_future,
                            proprio_dim,
                            proprio_weight,
                        )
                    )
                    reconstruction_rows.append(
                        float((0.5 * (total_t + total_future)).cpu())
                    )
                    dino_rows.append(float((0.5 * (dino_t + dino_future)).cpu()))
                    proprio_rows.append(
                        float((0.5 * (proprio_t + proprio_future)).cpu())
                    )
                latent_rows.append(z_t.cpu().numpy())
                latent_rows.append(z_future.cpu().numpy())
        latents = np.concatenate(latent_rows)
        latent_std = latents.std(axis=0)
        return {
            "prediction_mse": float(np.mean(prediction_rows)),
            "reconstruction_mse": (
                float(np.mean(reconstruction_rows))
                if reconstruction_rows
                else 0.0
            ),
            "dino_reconstruction_mse": (
                float(np.mean(dino_rows)) if dino_rows else 0.0
            ),
            "proprio_reconstruction_mse": (
                float(np.mean(proprio_rows)) if proprio_rows else 0.0
            ),
            "variance_loss": float(np.mean(variance_rows)),
            "covariance_loss": float(np.mean(covariance_rows)),
            "kl_total": 0.0,
            "kl_per_dimension": 0.0,
            "active_dimensions_std_gt_0_1": int(np.sum(latent_std > 0.1)),
            "posterior_variance_mean": 0.0,
            "latent_norm_mean": float(np.linalg.norm(latents, axis=-1).mean()),
            "latent_norm_std": float(np.linalg.norm(latents, axis=-1).std()),
        }

    for epoch in trange(1, epochs + 1, desc=f"train predictive {candidate}"):
        encoder.train()
        predictor.train()
        target_encoder.eval()
        if decoder is not None:
            decoder.train()
        sums = {
            "loss": 0.0,
            "prediction": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
            "reconstruction": 0.0,
        }
        count = 0
        for batch in train_loader:
            x_t = batch["x_t"].to(device, non_blocking=True).float()
            x_future = batch["x_future"].to(device, non_blocking=True).float()
            actions = batch["actions"].to(device, non_blocking=True).float()
            horizon = batch["horizon"].to(device, non_blocking=True)
            z_t = encoder(x_t)
            z_future = encoder(x_future)
            with torch.no_grad():
                target = target_encoder(x_future)
            prediction = predictor(z_t, actions, horizon)
            prediction_loss = torch.mean((prediction - target) ** 2)
            variance_t, covariance_t = _variance_covariance_losses(z_t)
            variance_future, covariance_future = (
                _variance_covariance_losses(z_future)
            )
            variance_loss = 0.5 * (variance_t + variance_future)
            covariance_loss = 0.5 * (covariance_t + covariance_future)
            reconstruction_loss = torch.zeros((), device=device)
            if decoder is not None:
                reconstruction_t = decoder(z_t)
                reconstruction_future = decoder(z_future)
                total_t, _dino_t, _proprio_t = _reconstruction_loss(
                    reconstruction_t, x_t, proprio_dim, proprio_weight
                )
                total_future, _dino_future, _proprio_future = (
                    _reconstruction_loss(
                        reconstruction_future,
                        x_future,
                        proprio_dim,
                        proprio_weight,
                    )
                )
                reconstruction_loss = 0.5 * (total_t + total_future)
            loss = (
                lambda_pred * prediction_loss
                + lambda_var * variance_loss
                + lambda_cov * covariance_loss
                + lambda_recon * reconstruction_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_target()
            batch_count = len(x_t)
            sums["loss"] += float(loss.detach().cpu()) * batch_count
            sums["prediction"] += (
                float(prediction_loss.detach().cpu()) * batch_count
            )
            sums["variance"] += (
                float(variance_loss.detach().cpu()) * batch_count
            )
            sums["covariance"] += (
                float(covariance_loss.detach().cpu()) * batch_count
            )
            sums["reconstruction"] += (
                float(reconstruction_loss.detach().cpu()) * batch_count
            )
            count += batch_count
        encoder.eval()
        predictor.eval()
        if decoder is not None:
            decoder.eval()
        metrics = validation_metrics()
        selection = (
            lambda_pred * metrics["prediction_mse"]
            + lambda_var * metrics["variance_loss"]
            + lambda_cov * metrics["covariance_loss"]
            + lambda_recon * metrics["reconstruction_mse"]
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value / count for key, value in sums.items()},
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        if selection < best_validation:
            best_validation = selection
            epochs_without_improvement = 0
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "target_encoder": copy.deepcopy(target_encoder.state_dict()),
                "predictor": copy.deepcopy(predictor.state_dict()),
                "decoder": (
                    copy.deepcopy(decoder.state_dict())
                    if decoder is not None
                    else None
                ),
                "epoch": epoch,
            }
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Predictive representation produced no checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    target_encoder.load_state_dict(best_state["target_encoder"])
    predictor.load_state_dict(best_state["predictor"])
    if decoder is not None and best_state["decoder"] is not None:
        decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    if decoder is not None:
        decoder.eval()
    final_validation = validation_metrics()
    payload = {
        "candidate": candidate,
        "family": spec["family"],
        "spec": spec,
        "encoder_type": "deterministic",
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": width,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "world_model": predictor.state_dict(),
        "decoder": decoder.state_dict() if decoder is not None else None,
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "best_epoch": best_state["epoch"],
        "trained_epochs": len(history),
        "validation_metrics": final_validation,
        "history": history,
        "data": data_metadata,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    _write_representation_metrics(
        artifact_dir / "representation_metrics.json", payload
    )
    console.print(f"Wrote learned-interface representation: {checkpoint_path}")
    return checkpoint_path


def _encode_physical_labels(labels: np.ndarray) -> np.ndarray:
    yaw = labels[:, 2]
    return np.concatenate(
        [
            labels[:, :2],
            np.sin(yaw)[:, None],
            np.cos(yaw)[:, None],
            labels[:, 3:],
        ],
        axis=-1,
    ).astype(np.float32)


def _effect_teacher_path(config: Config, seed: int) -> Path:
    return (
        ensure_dir(
            config.path_value("paths.incremental_artifact_dir")
            / "learned_interface"
            / "effect_auxiliary_teacher"
        )
        / f"seed{seed}.pt"
    )


def _train_effect_auxiliary_teacher(
    config: Config,
    frame_norm: Standardizer,
    input_dim: int,
    seed: int,
    force: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    path = _effect_teacher_path(config, seed)
    device = default_device()
    if path.exists() and not force:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = MLP(
            input_dim,
            int(checkpoint["output_dim"]),
            int(checkpoint["hidden_dim"]),
            depth=3,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, checkpoint

    set_seed(seed + 17)
    probe_path = collect_phase6_probe_dataset(config, force=False)
    with np.load(probe_path) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        labels = _encode_physical_labels(
            np.asarray(data["labels"], dtype=np.float32)
        )
        contact = np.asarray(data["contact"], dtype=np.float32)
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(len(inputs))
    split = int(0.8 * len(order))
    train_idx, validation_idx = order[:split], order[split:]
    label_norm = Standardizer.fit(labels[train_idx])
    x = frame_norm.transform(inputs)
    y = label_norm.transform(labels)
    hidden_dim = int(
        config.get("learned_interface.effect.teacher_hidden_dim", 512)
    )
    model = MLP(input_dim, y.shape[-1] + 1, hidden_dim, depth=3).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learned_interface.effect.teacher_lr", 3e-4)),
    )
    batch_size = int(
        config.get("learned_interface.effect.teacher_batch_size", 512)
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x[train_idx]),
            torch.from_numpy(y[train_idx]),
            torch.from_numpy(contact[train_idx]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    epochs = int(config.get("learned_interface.effect.teacher_epochs", 100))
    for _epoch in trange(epochs, desc="train effect auxiliary teacher"):
        model.train()
        for batch_x, batch_y, batch_contact in loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()
            batch_contact = batch_contact.to(
                device, non_blocking=True
            ).float()
            prediction = model(batch_x)
            loss = torch.mean((prediction[:, :-1] - batch_y) ** 2)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[:, -1:], batch_contact
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(
                torch.from_numpy(x[validation_idx]).to(device).float()
            )
            validation_loss = torch.mean(
                (
                    validation_prediction[:, :-1]
                    - torch.from_numpy(y[validation_idx]).to(device).float()
                )
                ** 2
            )
            validation_loss = (
                validation_loss
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    validation_prediction[:, -1:],
                    torch.from_numpy(contact[validation_idx])
                    .to(device)
                    .float(),
                )
            )
        score = float(validation_loss.cpu())
        if score < best_loss:
            best_loss = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Effect auxiliary teacher produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = (
            model(torch.from_numpy(x[validation_idx]).to(device).float())
            .cpu()
            .numpy()
        )
    decoded = label_norm.inverse(prediction[:, :-1])
    target = labels[validation_idx]
    yaw_prediction = np.arctan2(decoded[:, 2], decoded[:, 3])
    yaw_target = np.arctan2(target[:, 2], target[:, 3])
    yaw_error = np.arctan2(
        np.sin(yaw_prediction - yaw_target),
        np.cos(yaw_prediction - yaw_target),
    )
    contact_logits = prediction[:, -1]
    contact_target = contact[validation_idx, 0]
    checkpoint = {
        "model": model.state_dict(),
        "input_dim": input_dim,
        "output_dim": int(y.shape[-1] + 1),
        "hidden_dim": hidden_dim,
        "frame_norm": frame_norm.state_dict(),
        "label_norm": label_norm.state_dict(),
        "probe_dataset": str(probe_path),
        "validation": {
            "normalized_loss": best_loss,
            "object_position_rmse_m": float(
                np.sqrt(np.mean((decoded[:, :2] - target[:, :2]) ** 2))
            ),
            "object_yaw_mae_rad": float(np.mean(np.abs(yaw_error))),
            "tcp_position_rmse_m": float(
                np.sqrt(np.mean((decoded[:, 7:9] - target[:, 7:9]) ** 2))
            ),
            "contact_accuracy": float(
                np.mean((contact_logits >= 0.0) == contact_target.astype(bool))
            ),
            "contact_auroc": _binary_auc(contact_logits, contact_target),
        },
        "semantics": (
            "Frozen observation-to-physical-state probe trained on the "
            "causal Phase 6 probe dataset; its outputs are pseudo-labels, "
            "not privileged inputs to the deployed hierarchy."
        ),
    }
    torch.save(checkpoint, path)
    write_json(
        path.with_suffix(".json"),
        {key: value for key, value in checkpoint.items() if key != "model"},
    )
    return model, checkpoint


@torch.inference_mode()
def _effect_auxiliary_predictions(
    model: nn.Module,
    frame_norm: Standardizer,
    frames: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = frame_norm.transform(frames)
    rows = []
    for start in range(0, len(normalized), 2048):
        prediction = model(
            torch.from_numpy(normalized[start : start + 2048])
            .to(device)
            .float()
        )
        rows.append(
            torch.cat(
                [prediction[:, :-1], prediction[:, -1:].sigmoid()], dim=-1
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(rows).astype(np.float32)


def _train_effect_representation(
    config: Config,
    candidate: str,
    spec: dict[str, Any],
    seed: int,
    force: bool,
) -> Path:
    artifact_dir = _artifact_dir(config, candidate, seed)
    checkpoint_path = artifact_dir / "representation.pt"
    if checkpoint_path.exists() and not force:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        _write_representation_metrics(
            artifact_dir / "representation_metrics.json", checkpoint
        )
        console.print(f"Learned-interface representation exists: {checkpoint_path}")
        return checkpoint_path
    set_seed(seed)
    train_episodes, validation_episodes, data_metadata = (
        _load_phase6_train_episodes(config)
    )
    frame_norm, action_norm = _phase4_fit_standardizers(train_episodes)
    input_dim = int(data_metadata["frame_dim"])
    effect_dim = int(spec["latent_dim"])
    width = int(spec["width"])
    horizon_steps = int(config.get("learned_interface.horizon_steps", 10))
    device = default_device()
    auxiliary_teacher, teacher_checkpoint = (
        _train_effect_auxiliary_teacher(
            config, frame_norm, input_dim, seed, force=False
        )
    )

    def normalized(
        episodes: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        zero_action = action_norm.transform(
            np.zeros((1, 3), dtype=np.float32)
        )[0]
        return [
            {
                "frames": frame_norm.transform(episode["frames"]),
                "actions": action_norm.transform(episode["actions"]),
                "zero_action": zero_action,
                "auxiliary": _effect_auxiliary_predictions(
                    auxiliary_teacher,
                    frame_norm,
                    episode["frames"],
                    device,
                ),
            }
            for episode in episodes
        ]

    train = normalized(train_episodes)
    validation = normalized(validation_episodes)
    encoder = _EffectEncoder(input_dim, effect_dim, width).to(device)
    action_head = MLP(
        input_dim + effect_dim + 4, 3, width, depth=4
    ).to(device)
    auxiliary_head = MLP(effect_dim, 12, width, depth=3).to(device)
    optimizer = torch.optim.AdamW(
        [
            *encoder.parameters(),
            *action_head.parameters(),
            *auxiliary_head.parameters(),
        ],
        lr=float(config.get("learned_interface.representation.lr", 3e-4)),
    )
    batch_size = int(config.get("learned_interface.representation.batch_size", 512))
    batches_per_epoch = int(
        spec.get(
            "batches_per_epoch",
            config.get("learned_interface.representation.batches_per_epoch", 400),
        )
    )
    train_loader = DataLoader(
        _EffectRepresentationDataset(
            train,
            horizon_steps,
            batch_size * batches_per_epoch,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        _EffectRepresentationDataset(
            validation,
            horizon_steps,
            int(
                config.get(
                    "learned_interface.representation.validation_samples",
                    8192,
                )
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    lambda_action = float(spec["lambda_action"])
    lambda_auxiliary = float(spec["lambda_auxiliary"])
    lambda_var = float(spec["lambda_var"])
    lambda_cov = float(spec["lambda_cov"])
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    patience = int(spec["early_stopping_patience"])
    epochs_without_improvement = 0
    timer = Timer()

    def losses(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        x_start = batch["x_start"].to(device).float()
        x_future = batch["x_future"].to(device).float()
        x_current = batch["x_current"].to(device).float()
        previous = batch["previous"].to(device).float()
        remaining = batch["remaining"].to(device).float()[:, None]
        target_action = batch["action"].to(device).float()
        target_auxiliary = batch["auxiliary"].to(device).float()
        pair = torch.cat(
            [x_start, x_future, torch.ones_like(remaining)], dim=-1
        )
        effect = encoder(pair)
        action_prediction = action_head(
            torch.cat(
                [x_current, effect, previous, remaining], dim=-1
            )
        )
        auxiliary_prediction = auxiliary_head(effect)
        action_loss = torch.mean((action_prediction - target_action) ** 2)
        auxiliary_continuous = torch.mean(
            (auxiliary_prediction[:, :-1] - target_auxiliary[:, :-1]) ** 2
        )
        auxiliary_contact = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                auxiliary_prediction[:, -1:],
                target_auxiliary[:, -1:],
            )
        )
        variance, covariance = _variance_covariance_losses(effect)
        total = (
            lambda_action * action_loss
            + lambda_auxiliary * (auxiliary_continuous + auxiliary_contact)
            + lambda_var * variance
            + lambda_cov * covariance
        )
        return (
            total,
            {
                "action_mse": action_loss,
                "auxiliary_continuous_mse": auxiliary_continuous,
                "auxiliary_contact_bce": auxiliary_contact,
                "variance_loss": variance,
                "covariance_loss": covariance,
            },
            effect,
        )

    def validation_metrics() -> dict[str, Any]:
        rows: dict[str, list[float]] = {
            "action_mse": [],
            "auxiliary_continuous_mse": [],
            "auxiliary_contact_bce": [],
            "variance_loss": [],
            "covariance_loss": [],
        }
        effects = []
        with torch.inference_mode():
            for batch in validation_loader:
                _total, metrics, effect = losses(batch)
                for key, value in metrics.items():
                    rows[key].append(float(value.cpu()))
                effects.append(effect.cpu().numpy())
        effect_array = np.concatenate(effects)
        effect_std = effect_array.std(axis=0)
        result = {key: float(np.mean(value)) for key, value in rows.items()}
        result.update(
            {
                "active_dimensions_std_gt_0_1": int(
                    np.sum(effect_std > 0.1)
                ),
                "latent_norm_mean": float(
                    np.linalg.norm(effect_array, axis=-1).mean()
                ),
                "latent_norm_std": float(
                    np.linalg.norm(effect_array, axis=-1).std()
                ),
                "reconstruction_mse": 0.0,
                "kl_total": 0.0,
                "kl_per_dimension": 0.0,
                "posterior_variance_mean": 0.0,
            }
        )
        return result

    epochs = int(
        spec.get(
            "epochs",
            config.get("learned_interface.representation.epochs", 60),
        )
    )
    for epoch in trange(1, epochs + 1, desc=f"train effect {candidate}"):
        encoder.train()
        action_head.train()
        auxiliary_head.train()
        sums: dict[str, float] = {}
        count = 0
        for batch in train_loader:
            total, metrics, _effect = losses(batch)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            batch_count = len(batch["x_start"])
            sums["loss"] = sums.get("loss", 0.0) + float(total.cpu()) * batch_count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.cpu()) * batch_count
            count += batch_count
        encoder.eval()
        action_head.eval()
        auxiliary_head.eval()
        validation_row = validation_metrics()
        selection = (
            lambda_action * validation_row["action_mse"]
            + lambda_auxiliary
            * (
                validation_row["auxiliary_continuous_mse"]
                + validation_row["auxiliary_contact_bce"]
            )
            + lambda_var * validation_row["variance_loss"]
            + lambda_cov * validation_row["covariance_loss"]
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value / count for key, value in sums.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_row.items()
                },
            }
        )
        if selection < best_validation:
            best_validation = selection
            epochs_without_improvement = 0
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "action_head": copy.deepcopy(action_head.state_dict()),
                "auxiliary_head": copy.deepcopy(auxiliary_head.state_dict()),
                "epoch": epoch,
            }
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Effect representation produced no checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    action_head.load_state_dict(best_state["action_head"])
    auxiliary_head.load_state_dict(best_state["auxiliary_head"])
    encoder.eval()
    action_head.eval()
    auxiliary_head.eval()
    payload = {
        "candidate": candidate,
        "family": spec["family"],
        "spec": spec,
        "encoder_type": "effect",
        "input_dim": input_dim,
        "latent_dim": effect_dim,
        "hidden_dim": width,
        "encoder": encoder.state_dict(),
        "action_head": action_head.state_dict(),
        "auxiliary_head": auxiliary_head.state_dict(),
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "horizon_steps": horizon_steps,
        "best_epoch": best_state["epoch"],
        "trained_epochs": len(history),
        "validation_metrics": validation_metrics(),
        "auxiliary_teacher": {
            "checkpoint": str(_effect_teacher_path(config, seed)),
            "validation": teacher_checkpoint["validation"],
            "semantics": teacher_checkpoint["semantics"],
        },
        "history": history,
        "data": data_metadata,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    _write_representation_metrics(
        artifact_dir / "representation_metrics.json", payload
    )
    console.print(f"Wrote learned-interface representation: {checkpoint_path}")
    return checkpoint_path


def train_learned_interface_representation(
    config: Config,
    candidate: str,
    seed: int = 0,
    force: bool = False,
) -> Path:
    spec = learned_interface_candidate_spec(config, candidate)
    representation_candidate = str(spec["representation_candidate"])
    if representation_candidate != candidate:
        return train_learned_interface_representation(
            config, representation_candidate, seed, force=force
        )
    if str(spec["family"]) == "predictive_jepa":
        return _train_predictive_representation(
            config, candidate, spec, seed, force
        )
    if str(spec["family"]) == "effect_code":
        return _train_effect_representation(
            config, candidate, spec, seed, force
        )
    artifact_dir = _artifact_dir(config, candidate, seed)
    checkpoint_path = artifact_dir / "representation.pt"
    if checkpoint_path.exists() and not force:
        metrics_path = artifact_dir / "representation_metrics.json"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        _write_representation_metrics(metrics_path, checkpoint)
        console.print(f"Learned-interface representation exists: {checkpoint_path}")
        return checkpoint_path
    set_seed(seed)
    train_episodes, validation_episodes, data_metadata = _load_phase6_train_episodes(
        config
    )
    frame_norm, _action_norm = _phase4_fit_standardizers(train_episodes)

    def normalized(
        episodes: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        return [
            {
                "frames": frame_norm.transform(episode["frames"]),
                "actions": episode["actions"],
            }
            for episode in episodes
        ]

    train = normalized(train_episodes)
    validation = normalized(validation_episodes)
    input_dim = int(data_metadata["frame_dim"])
    latent_dim = int(spec["latent_dim"])
    width = int(spec["width"])
    encoder_type = str(spec["encoder_type"])
    device = default_device()
    if encoder_type == "vae":
        encoder: nn.Module = VariationalObservationEncoder(
            input_dim, latent_dim, width
        ).to(device)
    elif encoder_type == "deterministic":
        encoder = ObservationEncoder(input_dim, latent_dim, width).to(device)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    decoder = MLP(latent_dim, input_dim, width, depth=3).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(config.get("learned_interface.representation.lr", 3e-4)),
    )
    batch_size = int(config.get("learned_interface.representation.batch_size", 512))
    batches_per_epoch = int(
        config.get("learned_interface.representation.batches_per_epoch", 400)
    )
    epochs = int(config.get("learned_interface.representation.epochs", 60))
    dataset = _RepresentationDataset(
        train, length=batch_size * batches_per_epoch
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_samples = int(
        config.get("learned_interface.representation.validation_samples", 8192)
    )
    validation_dataset = _RepresentationDataset(
        validation, length=validation_samples
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    beta = float(spec["beta"])
    warmup_steps = int(spec["kl_warmup_steps"])
    free_bits = float(spec["free_bits"])
    dino_noise = float(spec["dino_noise_std"])
    proprio_noise = float(spec["proprio_noise_std"])
    reconstruction_weight = float(spec["reconstruction_weight"])
    proprio_dim = int(config.get("incremental.phase6.proprio_dim", 21))
    proprio_weight = float(
        config.get("incremental.phase6.proprio_reconstruction_weight", 1.0)
    )
    best_state: dict[str, Any] | None = None
    best_validation = float("inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    timer = Timer()

    def add_noise(x: torch.Tensor) -> torch.Tensor:
        if dino_noise <= 0.0 and proprio_noise <= 0.0:
            return x
        corrupted = x.clone()
        if dino_noise > 0.0:
            corrupted[:, :-proprio_dim] += torch.randn_like(
                corrupted[:, :-proprio_dim]
            ) * dino_noise
        if proprio_noise > 0.0:
            corrupted[:, -proprio_dim:] += torch.randn_like(
                corrupted[:, -proprio_dim:]
            ) * proprio_noise
        return corrupted

    def encode_training(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_type == "vae":
            assert isinstance(encoder, VariationalObservationEncoder)
            z, mean, logvar = encoder.sample(x)
            kl_total, kl_per_dim = _vae_kl(mean, logvar, free_bits)
            return z, kl_total, kl_per_dim
        return (
            encoder(x),
            torch.zeros((), device=x.device),
            torch.zeros((), device=x.device),
        )

    def validation_metrics() -> dict[str, Any]:
        recon_total = []
        recon_dino = []
        recon_proprio = []
        kl_total = []
        kl_per_dim = []
        posterior_variance = []
        latent_rows = []
        with torch.inference_mode():
            for x in validation_loader:
                x = x.to(device).float()
                if encoder_type == "vae":
                    assert isinstance(encoder, VariationalObservationEncoder)
                    mean, logvar = encoder.encode_stats(x)
                    z = mean
                    total, per_dim = _vae_kl(mean, logvar, 0.0)
                    kl_total.append(float(total.cpu()))
                    kl_per_dim.append(float(per_dim.cpu()))
                    posterior_variance.append(
                        float(logvar.exp().mean().cpu())
                    )
                else:
                    z = encoder(x)
                reconstruction = decoder(z)
                total, dino, proprio = _reconstruction_loss(
                    reconstruction, x, proprio_dim, proprio_weight
                )
                recon_total.append(float(total.cpu()))
                recon_dino.append(float(dino.cpu()))
                recon_proprio.append(float(proprio.cpu()))
                latent_rows.append(z.cpu().numpy())
        latents = np.concatenate(latent_rows)
        latent_std = latents.std(axis=0)
        return {
            "reconstruction_mse": float(np.mean(recon_total)),
            "dino_reconstruction_mse": float(np.mean(recon_dino)),
            "proprio_reconstruction_mse": float(np.mean(recon_proprio)),
            "kl_total": float(np.mean(kl_total)) if kl_total else 0.0,
            "kl_per_dimension": float(np.mean(kl_per_dim)) if kl_per_dim else 0.0,
            "active_dimensions_std_gt_0_1": int(np.sum(latent_std > 0.1)),
            "posterior_variance_mean": (
                float(np.mean(posterior_variance))
                if posterior_variance
                else 0.0
            ),
            "latent_norm_mean": float(np.linalg.norm(latents, axis=-1).mean()),
            "latent_norm_std": float(np.linalg.norm(latents, axis=-1).std()),
        }

    for epoch in trange(
        1, epochs + 1, desc=f"train learned interface {candidate}"
    ):
        encoder.train()
        decoder.train()
        sums = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "dino": 0.0,
            "proprio": 0.0,
            "kl_total": 0.0,
            "kl_per_dimension": 0.0,
            "beta": 0.0,
        }
        count = 0
        for x in loader:
            x = x.to(device, non_blocking=True).float()
            z, kl_total, kl_per_dim = encode_training(add_noise(x))
            reconstruction = decoder(z)
            recon, dino, proprio = _reconstruction_loss(
                reconstruction, x, proprio_dim, proprio_weight
            )
            beta_scale = (
                min(1.0, global_step / warmup_steps)
                if warmup_steps > 0
                else 1.0
            )
            effective_beta = beta * beta_scale
            loss = reconstruction_weight * recon + effective_beta * kl_total
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = len(x)
            sums["loss"] += float(loss.detach().cpu()) * batch_count
            sums["reconstruction"] += float(recon.detach().cpu()) * batch_count
            sums["dino"] += float(dino.detach().cpu()) * batch_count
            sums["proprio"] += float(proprio.detach().cpu()) * batch_count
            sums["kl_total"] += float(kl_total.detach().cpu()) * batch_count
            sums["kl_per_dimension"] += (
                float(kl_per_dim.detach().cpu()) * batch_count
            )
            sums["beta"] += effective_beta * batch_count
            count += batch_count
            global_step += 1
        encoder.eval()
        decoder.eval()
        validation_metrics_row = validation_metrics()
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value / count for key, value in sums.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics_row.items()
                },
            }
        )
        selection = float(validation_metrics_row["reconstruction_mse"])
        if selection < best_validation:
            best_validation = selection
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "decoder": copy.deepcopy(decoder.state_dict()),
                "epoch": epoch,
            }
    if best_state is None:
        raise RuntimeError("Learned-interface representation produced no checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval()
    decoder.eval()
    final_validation = validation_metrics()
    payload = {
        "candidate": candidate,
        "family": spec["family"],
        "spec": spec,
        "encoder_type": encoder_type,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": width,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "frame_norm": frame_norm.state_dict(),
        "best_epoch": best_state["epoch"],
        "validation_metrics": final_validation,
        "history": history,
        "data": data_metadata,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    _write_representation_metrics(
        artifact_dir / "representation_metrics.json",
        payload,
    )
    console.print(f"Wrote learned-interface representation: {checkpoint_path}")
    return checkpoint_path


def _load_representation(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint["encoder_type"] == "effect":
        encoder: nn.Module = _EffectEncoder(
            int(checkpoint["input_dim"]),
            int(checkpoint["latent_dim"]),
            int(checkpoint["hidden_dim"]),
        ).to(device)
    else:
        encoder_cls = (
            VariationalObservationEncoder
            if checkpoint["encoder_type"] == "vae"
            else ObservationEncoder
        )
        encoder = encoder_cls(
            int(checkpoint["input_dim"]),
            int(checkpoint["latent_dim"]),
            int(checkpoint["hidden_dim"]),
        ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    return encoder, checkpoint


@torch.inference_mode()
def _encode_array(
    encoder: nn.Module,
    frame_norm: Standardizer,
    frames: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = frame_norm.transform(frames)
    rows = []
    for start in range(0, len(normalized), 2048):
        rows.append(
            encoder(
                torch.from_numpy(normalized[start : start + 2048])
                .to(device)
                .float()
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(rows).astype(np.float32)


@torch.inference_mode()
def _encode_effect_array(
    encoder: nn.Module,
    frame_norm: Standardizer,
    start_frames: np.ndarray,
    future_frames: np.ndarray,
    horizon_fraction: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    start = frame_norm.transform(start_frames)
    future = frame_norm.transform(future_frames)
    pair = np.concatenate(
        [start, future, horizon_fraction.reshape(-1, 1)], axis=-1
    ).astype(np.float32)
    rows = []
    for index in range(0, len(pair), 2048):
        rows.append(
            encoder(
                torch.from_numpy(pair[index : index + 2048])
                .to(device)
                .float()
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(rows).astype(np.float32)


def probe_learned_interface_representation(
    config: Config,
    candidate: str,
    seed: int = 0,
    force: bool = False,
) -> Path:
    output_path = _result_dir(config, candidate, seed) / "representation_probe.json"
    if output_path.exists() and not force:
        console.print(f"Learned-interface probe exists: {output_path}")
        return output_path
    checkpoint_path = train_learned_interface_representation(
        config, candidate, seed, force=False
    )
    device = default_device()
    encoder, checkpoint = _load_representation(checkpoint_path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    if checkpoint["encoder_type"] == "effect":
        payload = {
            "stage": "representation_probe",
            "candidate": candidate,
            "representation_checkpoint": str(checkpoint_path),
            "representation_validation": checkpoint["validation_metrics"],
            "auxiliary_teacher": checkpoint["auxiliary_teacher"],
            "probe_kind": (
                "Pairwise effect codes are evaluated by their action and "
                "future-physical-target auxiliary heads rather than the "
                "unary Phase 6 state probe."
            ),
            "metadata": _runtime_metadata(config),
        }
        write_json(output_path, payload)
        console.print(payload)
        return output_path
    dataset_path = collect_phase6_probe_dataset(config, force=False)
    with np.load(dataset_path) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        next_inputs = np.asarray(data["next_inputs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        next_labels = np.asarray(data["next_labels"], dtype=np.float32)
        contact = np.asarray(data["contact"], dtype=np.float32)
        reward = np.asarray(data["reward"], dtype=np.float32)
    representations = _encode_array(encoder, frame_norm, inputs, device)
    next_representations = _encode_array(
        encoder, frame_norm, next_inputs, device
    )
    metrics = _phase6_train_probe_heads(
        config,
        representations,
        next_representations,
        actions,
        labels,
        next_labels,
        contact,
        reward,
        seed,
    )
    payload = {
        "stage": "representation_probe",
        "candidate": candidate,
        "representation_checkpoint": str(checkpoint_path),
        "representation_validation": checkpoint["validation_metrics"],
        "probe_dataset": str(dataset_path),
        **metrics,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def prepare_learned_interface_episodes(
    config: Config,
    candidate: str,
    seed: int = 0,
    force: bool = False,
) -> Path:
    artifact_dir = _artifact_dir(config, candidate, seed)
    output_path = artifact_dir / "encoded_episodes.pt"
    if output_path.exists() and not force:
        console.print(f"Learned-interface encoded episodes exist: {output_path}")
        return output_path
    checkpoint_path = train_learned_interface_representation(
        config, candidate, seed, force=False
    )
    device = default_device()
    encoder, checkpoint = _load_representation(checkpoint_path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    train, validation, data_metadata = _load_phase6_train_episodes(config)
    horizon_steps = int(config.get("learned_interface.horizon_steps", 10))

    def convert(
        episodes: list[dict[str, np.ndarray]],
    ) -> list[np.ndarray]:
        if checkpoint["encoder_type"] != "effect":
            return [
                _encode_array(encoder, frame_norm, episode["frames"], device)
                for episode in episodes
            ]
        converted = []
        for episode in episodes:
            goals = np.zeros(
                (len(episode["frames"]), int(checkpoint["latent_dim"])),
                dtype=np.float32,
            )
            if len(goals) > horizon_steps:
                goals[horizon_steps:] = _encode_effect_array(
                    encoder,
                    frame_norm,
                    episode["frames"][:-horizon_steps],
                    episode["frames"][horizon_steps:],
                    np.ones(len(goals) - horizon_steps, dtype=np.float32),
                    device,
                )
            converted.append(goals)
        return converted

    payload = {
        "format_version": 2,
        "candidate": candidate,
        "representation_checkpoint": str(checkpoint_path),
        "train_goals": convert(train),
        "validation_goals": convert(validation),
        "data": data_metadata,
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    console.print(f"Wrote learned-interface encoded episodes: {output_path}")
    return output_path


class _HeldGoalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        frame_norm: Standardizer,
        goal_norm: Standardizer,
        action_norm: Standardizer,
        horizon_steps: int,
        mode: str,
        length: int,
        conditioning: str = "concat",
    ) -> None:
        if mode not in {"high", "low"}:
            raise ValueError(f"Unknown held-goal mode: {mode}")
        if conditioning not in {"concat", "delta", "relation", "film"}:
            raise ValueError(f"Unknown goal conditioning: {conditioning}")
        self.episodes = [
            episode
            for episode in episodes
            if len(episode["actions"]) > horizon_steps
        ]
        self.frame_norm = frame_norm
        self.goal_norm = goal_norm
        self.action_norm = action_norm
        self.horizon_steps = horizon_steps
        self.mode = mode
        self.length = length
        self.conditioning = conditioning
        self.zero_action = action_norm.transform(
            np.zeros((1, 3), dtype=np.float32)
        )[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        base = int(
            np.random.randint(0, len(episode["actions"]) - self.horizon_steps)
        )
        previous = (
            self.action_norm.transform(
                episode["actions"][base - 1 : base]
            )[0]
            if base > 0
            else self.zero_action
        )
        if self.mode == "high":
            condition = np.concatenate(
                [
                    self.frame_norm.transform(
                        episode["frames"][base : base + 1]
                    )[0],
                    previous,
                ]
            )
            target = self.goal_norm.transform(
                episode["goals"][
                    base + self.horizon_steps : base + self.horizon_steps + 1
                ]
            )[0]
        else:
            offset = int(np.random.randint(0, self.horizon_steps))
            current = base + offset
            previous = (
                self.action_norm.transform(
                    episode["actions"][current - 1 : current]
                )[0]
                if current > 0
                else self.zero_action
            )
            frame = self.frame_norm.transform(
                episode["frames"][current : current + 1]
            )
            current_goal = self.goal_norm.transform(
                episode["goals"][current : current + 1]
            )
            future_goal = self.goal_norm.transform(
                episode["goals"][
                    base
                    + self.horizon_steps : base
                    + self.horizon_steps
                    + 1
                ]
            )
            remaining = np.asarray(
                [[(self.horizon_steps - offset) / self.horizon_steps]],
                dtype=np.float32,
            )
            condition = _low_condition_array(
                frame,
                current_goal,
                future_goal,
                previous[None],
                remaining,
                self.conditioning,
            )[0]
            target = self.action_norm.transform(
                episode["actions"][current : current + 1]
            )[0]
        return (
            torch.from_numpy(condition.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


def _low_condition_array(
    frames: np.ndarray,
    current_goals: np.ndarray,
    future_goals: np.ndarray,
    previous_actions: np.ndarray,
    remaining: np.ndarray,
    conditioning: str,
) -> np.ndarray:
    if conditioning == "concat" or conditioning == "film":
        goal_features = future_goals
    elif conditioning == "delta":
        goal_features = future_goals - current_goals
    elif conditioning == "relation":
        goal_features = np.concatenate(
            [current_goals, future_goals], axis=-1
        )
    else:
        raise ValueError(f"Unknown goal conditioning: {conditioning}")
    return np.concatenate(
        [frames, goal_features, previous_actions, remaining], axis=-1
    ).astype(np.float32)


class _GoalConditionedLowPolicy(nn.Module):
    def __init__(
        self,
        frame_dim: int,
        goal_dim: int,
        hidden_dim: int,
        conditioning: str,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.conditioning = conditioning
        if conditioning in {"concat", "delta"}:
            self.policy = MLP(
                frame_dim + goal_dim + 4, 3, hidden_dim, depth=4
            )
        elif conditioning == "relation":
            self.relation = MLP(
                2 * goal_dim + 1, goal_dim, hidden_dim, depth=2
            )
            self.policy = MLP(
                frame_dim + goal_dim + 4, 3, hidden_dim, depth=4
            )
        elif conditioning == "film":
            self.input_layer = nn.Linear(frame_dim + 4, hidden_dim)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]
            )
            self.modulators = nn.ModuleList(
                [nn.Linear(goal_dim, 2 * hidden_dim) for _ in range(4)]
            )
            self.output_layer = nn.Linear(hidden_dim, 3)
            for modulator in self.modulators:
                nn.init.zeros_(modulator.weight)
                nn.init.zeros_(modulator.bias)
        else:
            raise ValueError(f"Unknown goal conditioning: {conditioning}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conditioning in {"concat", "delta"}:
            return self.policy(x)
        if self.conditioning == "relation":
            frame = x[:, : self.frame_dim]
            goals = x[
                :, self.frame_dim : self.frame_dim + 2 * self.goal_dim
            ]
            tail = x[:, self.frame_dim + 2 * self.goal_dim :]
            relation = self.relation(
                torch.cat([goals, tail[:, -1:]], dim=-1)
            )
            return self.policy(torch.cat([frame, relation, tail], dim=-1))
        frame = x[:, : self.frame_dim]
        goal = x[
            :, self.frame_dim : self.frame_dim + self.goal_dim
        ]
        tail = x[:, self.frame_dim + self.goal_dim :]
        base = torch.cat([frame, tail], dim=-1)
        layers = [self.input_layer, *self.hidden_layers]
        h = base
        for layer, modulator in zip(layers, self.modulators, strict=True):
            h = torch.nn.functional.silu(layer(h))
            gamma, beta = modulator(goal).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        return self.output_layer(h)


@torch.inference_mode()
def _hierarchy_validation_metrics(
    high_model: nn.Module,
    low_model: nn.Module,
    episodes: list[dict[str, np.ndarray]],
    frame_norm: Standardizer,
    goal_norm: Standardizer,
    action_norm: Standardizer,
    horizon_steps: int,
    samples: int,
    seed: int,
    conditioning: str,
) -> dict[str, float]:
    device = next(high_model.parameters()).device
    rng = np.random.default_rng(seed)
    zero_action = action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
    high_conditions = []
    oracle_goals = []
    current_frames = []
    current_goals = []
    previous_actions = []
    target_actions = []
    remaining_values = []
    for _ in range(samples):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        base = int(rng.integers(0, len(episode["actions"]) - horizon_steps))
        offset = int(rng.integers(0, horizon_steps))
        current = base + offset
        high_previous = (
            action_norm.transform(episode["actions"][base - 1 : base])[0]
            if base > 0
            else zero_action
        )
        previous = (
            action_norm.transform(
                episode["actions"][current - 1 : current]
            )[0]
            if current > 0
            else zero_action
        )
        high_conditions.append(
            np.concatenate(
                [
                    frame_norm.transform(
                        episode["frames"][base : base + 1]
                    )[0],
                    high_previous,
                ]
            )
        )
        oracle_goals.append(
            goal_norm.transform(
                episode["goals"][
                    base + horizon_steps : base + horizon_steps + 1
                ]
            )[0]
        )
        current_frames.append(
            frame_norm.transform(
                episode["frames"][current : current + 1]
            )[0]
        )
        current_goals.append(
            goal_norm.transform(
                episode["goals"][current : current + 1]
            )[0]
        )
        previous_actions.append(previous)
        target_actions.append(episode["actions"][current])
        remaining_values.append((horizon_steps - offset) / horizon_steps)
    high_conditions_np = np.asarray(high_conditions, dtype=np.float32)
    oracle_goals_np = np.asarray(oracle_goals, dtype=np.float32)
    predicted_goals = high_model(
        torch.from_numpy(high_conditions_np).to(device).float()
    ).cpu().numpy()

    def low_actions(goals: np.ndarray) -> np.ndarray:
        condition = _low_condition_array(
            np.asarray(current_frames, dtype=np.float32),
            np.asarray(current_goals, dtype=np.float32),
            goals,
            np.asarray(previous_actions, dtype=np.float32),
            np.asarray(remaining_values, dtype=np.float32)[:, None],
            conditioning,
        )
        return action_norm.inverse(
            low_model(torch.from_numpy(condition).to(device).float())
            .cpu()
            .numpy()
        )

    target_actions_np = np.asarray(target_actions, dtype=np.float32)
    oracle_actions = low_actions(oracle_goals_np)
    predicted_actions = low_actions(predicted_goals)
    goal_errors = np.linalg.norm(predicted_goals - oracle_goals_np, axis=-1)
    return {
        "normalized_goal_l2": float(np.mean(goal_errors)),
        "oracle_action_mae": float(
            np.mean(np.abs(oracle_actions - target_actions_np))
        ),
        "predicted_action_mae": float(
            np.mean(np.abs(predicted_actions - target_actions_np))
        ),
        "prediction_induced_action_l2": float(
            np.mean(np.linalg.norm(predicted_actions - oracle_actions, axis=-1))
        ),
    }


def train_learned_interface_hierarchy(
    config: Config,
    candidate: str,
    seed: int = 0,
    force: bool = False,
) -> Path:
    artifact_dir = _artifact_dir(config, candidate, seed)
    checkpoint_path = artifact_dir / "hierarchy.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Learned-interface hierarchy exists: {checkpoint_path}")
        return checkpoint_path
    set_seed(seed)
    encoded_path = prepare_learned_interface_episodes(
        config, candidate, seed, force=False
    )
    encoded = torch.load(encoded_path, map_location="cpu", weights_only=False)
    representation_checkpoint = torch.load(
        encoded["representation_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if int(encoded.get("format_version", 1)) == 2:
        train_frames, validation_frames, _data_metadata = (
            _load_phase6_train_episodes(config)
        )

        def combine(
            frame_episodes: list[dict[str, np.ndarray]],
            goal_episodes: list[np.ndarray],
        ) -> list[dict[str, np.ndarray]]:
            if len(frame_episodes) != len(goal_episodes):
                raise ValueError("Learned-interface goal cache episode mismatch")
            return [
                {
                    "frames": frame_episode["frames"],
                    "goals": goals,
                    "actions": frame_episode["actions"],
                }
                for frame_episode, goals in zip(
                    frame_episodes, goal_episodes, strict=True
                )
            ]

        train = combine(train_frames, encoded["train_goals"])
        validation = combine(
            validation_frames, encoded["validation_goals"]
        )
    else:
        train = encoded["train"]
        validation = encoded["validation"]
    frame_norm = Standardizer.from_state_dict(
        representation_checkpoint["frame_norm"]
    )
    horizon_steps = int(config.get("learned_interface.horizon_steps", 10))
    goal_rows = (
        [
            episode["goals"][horizon_steps:]
            for episode in train
        ]
        if representation_checkpoint["encoder_type"] == "effect"
        else [episode["goals"] for episode in train]
    )
    goal_norm = Standardizer.fit(
        np.concatenate(goal_rows, axis=0)
    )
    action_norm = Standardizer.fit(
        np.concatenate([episode["actions"] for episode in train], axis=0)
    )
    conditioning = str(
        learned_interface_candidate_spec(config, candidate)["conditioning"]
    )
    batch_size = int(config.get("learned_interface.policy.batch_size", 512))
    batches_per_epoch = int(
        config.get("learned_interface.policy.batches_per_epoch", 200)
    )
    epochs = int(config.get("learned_interface.policy.epochs", 60))
    high_dataset = _HeldGoalDataset(
        train,
        frame_norm,
        goal_norm,
        action_norm,
        horizon_steps,
        "high",
        batch_size * batches_per_epoch,
        conditioning,
    )
    low_dataset = _HeldGoalDataset(
        train,
        frame_norm,
        goal_norm,
        action_norm,
        horizon_steps,
        "low",
        batch_size * batches_per_epoch,
        conditioning,
    )
    high_loader = DataLoader(
        high_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    low_loader = DataLoader(
        low_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    frame_dim = int(representation_checkpoint["input_dim"])
    goal_dim = int(representation_checkpoint["latent_dim"])
    hidden_dim = int(config.get("learned_interface.policy.hidden_dim", 512))
    device = default_device()
    high_model = MLP(frame_dim + 3, goal_dim, hidden_dim, depth=4).to(device)
    high_level_candidate = str(
        learned_interface_candidate_spec(config, candidate)[
            "high_level_candidate"
        ]
    )
    reused_high_level = high_level_candidate != candidate
    if reused_high_level:
        source_path = train_learned_interface_hierarchy(
            config, high_level_candidate, seed, force=False
        )
        source_checkpoint = torch.load(
            source_path, map_location="cpu", weights_only=False
        )
        if (
            int(source_checkpoint["frame_dim"]) != frame_dim
            or int(source_checkpoint["goal_dim"]) != goal_dim
        ):
            raise ValueError(
                "Reused high-level checkpoint has incompatible dimensions"
            )
        high_model.load_state_dict(source_checkpoint["high_model"])
        high_model.requires_grad_(False)
    low_model = _GoalConditionedLowPolicy(
        frame_dim, goal_dim, hidden_dim, conditioning
    ).to(device)
    learning_rate = float(config.get("learned_interface.policy.lr", 3e-4))
    high_optimizer = (
        None
        if reused_high_level
        else torch.optim.AdamW(high_model.parameters(), lr=learning_rate)
    )
    low_optimizer = torch.optim.AdamW(low_model.parameters(), lr=learning_rate)
    validation_samples = int(
        config.get("learned_interface.policy.validation_samples", 5000)
    )
    history = []
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    timer = Timer()
    for epoch in trange(
        1, epochs + 1, desc=f"train learned hierarchy {candidate}"
    ):
        if reused_high_level:
            high_model.eval()
        else:
            high_model.train()
        low_model.train()
        high_sum = 0.0
        low_sum = 0.0
        for (high_x, high_y), (low_x, low_y) in zip(
            high_loader, low_loader, strict=True
        ):
            if high_optimizer is not None:
                high_x = high_x.to(device, non_blocking=True)
                high_y = high_y.to(device, non_blocking=True)
                high_loss = torch.mean((high_model(high_x) - high_y) ** 2)
                high_optimizer.zero_grad(set_to_none=True)
                high_loss.backward()
                high_optimizer.step()
                high_sum += float(high_loss.detach().cpu())

            low_x = low_x.to(device, non_blocking=True)
            low_y = low_y.to(device, non_blocking=True)
            low_loss = torch.mean((low_model(low_x) - low_y) ** 2)
            low_optimizer.zero_grad(set_to_none=True)
            low_loss.backward()
            low_optimizer.step()
            low_sum += float(low_loss.detach().cpu())
        high_model.eval()
        low_model.eval()
        metrics = _hierarchy_validation_metrics(
            high_model,
            low_model,
            validation,
            frame_norm,
            goal_norm,
            action_norm,
            horizon_steps,
            validation_samples,
            seed + 3000,
            conditioning,
        )
        history.append(
            {
                "epoch": epoch,
                "high_train_mse": high_sum / batches_per_epoch,
                "low_train_mse": low_sum / batches_per_epoch,
                **metrics,
            }
        )
        if metrics["predicted_action_mae"] < best_score:
            best_score = metrics["predicted_action_mae"]
            best_state = {
                "high_model": copy.deepcopy(high_model.state_dict()),
                "low_model": copy.deepcopy(low_model.state_dict()),
                "validation": metrics,
                "epoch": epoch,
            }
    if best_state is None:
        raise RuntimeError("Learned-interface hierarchy produced no checkpoint")
    payload = {
        "candidate": candidate,
        "representation_checkpoint": encoded["representation_checkpoint"],
        "horizon_steps": horizon_steps,
        "update_period": int(config.get("learned_interface.update_period", 10)),
        "frame_dim": frame_dim,
        "goal_dim": goal_dim,
        "hidden_dim": hidden_dim,
        "conditioning": conditioning,
        "high_level_candidate": high_level_candidate,
        "high_model": best_state["high_model"],
        "low_model": best_state["low_model"],
        "frame_norm": frame_norm.state_dict(),
        "goal_norm": goal_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "best_epoch": best_state["epoch"],
        "validation_metrics": best_state["validation"],
        "history": history,
        "elapsed_s": timer.elapsed(),
        "data": encoded["data"],
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "hierarchy_metrics.json",
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "high_model",
                "low_model",
                "frame_norm",
                "goal_norm",
                "action_norm",
            }
        },
    )
    console.print(f"Wrote learned-interface hierarchy: {checkpoint_path}")
    return checkpoint_path


def _load_hierarchy(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    high_model = MLP(
        int(checkpoint["frame_dim"]) + 3,
        int(checkpoint["goal_dim"]),
        int(checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    low_model = _GoalConditionedLowPolicy(
        int(checkpoint["frame_dim"]),
        int(checkpoint["goal_dim"]),
        int(checkpoint["hidden_dim"]),
        str(checkpoint.get("conditioning", "concat")),
    ).to(device)
    high_model.load_state_dict(checkpoint["high_model"])
    low_state = checkpoint["low_model"]
    if "conditioning" not in checkpoint:
        low_state = {f"policy.{key}": value for key, value in low_state.items()}
    low_model.load_state_dict(low_state)
    high_model.eval()
    low_model.eval()
    return high_model, low_model


@torch.inference_mode()
def evaluate_learned_interface_hierarchy(
    config: Config,
    candidate: str,
    goal_source: str,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    if goal_source not in {"learned", "oracle"}:
        raise ValueError(f"Unknown goal source: {goal_source}")
    eval_episodes = int(
        episodes
        or (
            config.get("learned_interface.evaluation.oracle_episodes", 20)
            if goal_source == "oracle"
            else config.get("learned_interface.evaluation.screening_episodes", 20)
        )
    )
    output_path = (
        _result_dir(config, candidate, seed)
        / f"{goal_source}_hierarchy_eval_{eval_episodes}.json"
    )
    if output_path.exists() and not force:
        console.print(f"Learned-interface evaluation exists: {output_path}")
        return output_path
    hierarchy_path = train_learned_interface_hierarchy(
        config, candidate, seed, force=False
    )
    checkpoint = torch.load(
        hierarchy_path, map_location="cpu", weights_only=False
    )
    device = default_device()
    high_model, low_model = _load_hierarchy(checkpoint, device)
    encoder, representation_checkpoint = _load_representation(
        Path(checkpoint["representation_checkpoint"]), device
    )
    representation_frame_norm = Standardizer.from_state_dict(
        representation_checkpoint["frame_norm"]
    )
    is_effect = representation_checkpoint["encoder_type"] == "effect"
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    goal_norm = Standardizer.from_state_dict(checkpoint["goal_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    horizon_steps = int(checkpoint["horizon_steps"])
    update_period = int(checkpoint["update_period"])
    conditioning = str(checkpoint.get("conditioning", "concat"))
    max_steps = int(config.get("env_max_episode_steps", 100))
    max_num_envs = min(
        int(config.get("learned_interface.evaluation.num_envs", 16)),
        eval_episodes,
    )
    seed_start = int(
        config.get("learned_interface.evaluation.seed_start", 2_100_000)
    )
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    teacher_maes: list[float] = []
    goal_errors: list[float] = []
    replay_errors: list[float] = []
    high_decisions = 0
    progress = trange(
        eval_episodes, desc=f"eval {candidate} {goal_source}"
    )
    for batch_start in range(0, eval_episodes, max_num_envs):
        num_envs = min(max_num_envs, eval_episodes - batch_start)
        reset_seeds = [
            seed_start + batch_start + index for index in range(num_envs)
        ]
        student = gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )
        branch = (
            gym.make(
                config.get("env_id"),
                obs_mode="rgb+state",
                control_mode=config.get("control_mode"),
                reward_mode="normalized_dense",
                render_mode=None,
                sim_backend=_rl_backend(config),
                num_envs=num_envs,
                reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
            )
            if goal_source == "oracle"
            else None
        )
        action_low_np = np.asarray(student.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device)
        action_high = torch.as_tensor(action_high_np, device=device)
        zero_action = action_norm.transform(
            np.zeros((1, 3), dtype=np.float32)
        )[0]
        previous_action = np.repeat(zero_action[None], num_envs, axis=0)
        held_goal = np.zeros(
            (num_envs, int(checkpoint["goal_dim"])), dtype=np.float32
        )
        countdown = np.zeros(num_envs, dtype=np.int32)
        active = np.ones(num_envs, dtype=bool)
        success_once = np.zeros(num_envs, dtype=bool)
        batch_final = np.zeros(num_envs, dtype=np.float32)
        batch_max = np.full(num_envs, -np.inf, dtype=np.float32)
        history: list[torch.Tensor] = []
        try:
            obs, _info = student.reset(seed=reset_seeds)
            for _step in range(max_steps):
                if not np.any(active):
                    break
                frames = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                normalized_frames = frame_norm.transform(frames)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    predicted_goal = high_model(
                        torch.from_numpy(
                            np.concatenate(
                                [normalized_frames, previous_action], axis=-1
                            )
                        )
                        .to(device)
                        .float()
                    ).cpu().numpy()
                    selected_goal = predicted_goal
                    if branch is not None:
                        branch_obs, _branch_info = branch.reset(seed=reset_seeds)
                        for action_history in history:
                            branch_obs, _reward, _terminated, _truncated, _info = (
                                branch.step(action_history)
                            )
                        replay_error = torch.max(
                            torch.abs(
                                student.unwrapped.get_state()
                                - branch.unwrapped.get_state()
                            ),
                            dim=1,
                        ).values
                        replay_errors.extend(
                            replay_error.cpu().numpy()[replan].tolist()
                        )
                        for _ in range(horizon_steps):
                            branch_state = _phase7_obs_state_tensor(
                                branch_obs, device
                            )
                            branch_action = torch.clamp(
                                teacher.actor_mean(branch_state),
                                action_low,
                                action_high,
                            )
                            branch_obs, _reward, _terminated, _truncated, _info = (
                                branch.step(branch_action)
                            )
                        branch_frames = _phase4_frame_inputs(
                            branch_obs,
                            dino,
                            int(config.get("dino.batch_size", 64)),
                        )
                        oracle_encoding = (
                            _encode_effect_array(
                                encoder,
                                representation_frame_norm,
                                frames,
                                branch_frames,
                                np.ones(num_envs, dtype=np.float32),
                                device,
                            )
                            if is_effect
                            else _encode_array(
                                encoder,
                                representation_frame_norm,
                                branch_frames,
                                device,
                            )
                        )
                        oracle_goal = goal_norm.transform(oracle_encoding)
                        goal_errors.extend(
                            np.linalg.norm(
                                predicted_goal[replan] - oracle_goal[replan],
                                axis=-1,
                            ).tolist()
                        )
                        selected_goal = oracle_goal
                    held_goal[replan] = selected_goal[replan]
                    countdown[replan] = update_period
                    high_decisions += int(np.sum(replan))
                remaining = np.maximum(countdown, 1).astype(np.float32)
                if conditioning in {"delta", "relation"}:
                    if is_effect:
                        raise ValueError(
                            "Effect-code candidates require absolute concat "
                            "conditioning because a unary current effect is "
                            "undefined."
                        )
                    current_goal = goal_norm.transform(
                        _encode_array(
                            encoder,
                            representation_frame_norm,
                            frames,
                            device,
                        )
                    )
                else:
                    current_goal = np.empty_like(held_goal)
                condition = _low_condition_array(
                    normalized_frames,
                    current_goal,
                    held_goal,
                    previous_action,
                    (remaining / horizon_steps)[:, None],
                    conditioning,
                )
                raw_action = action_norm.inverse(
                    low_model(
                        torch.from_numpy(condition).to(device).float()
                    )
                    .cpu()
                    .numpy()
                )
                teacher_action = (
                    torch.clamp(
                        teacher.actor_mean(_phase7_obs_state_tensor(obs, device)),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                )
                teacher_maes.extend(
                    np.mean(
                        np.abs(raw_action[active] - teacher_action[active]),
                        axis=-1,
                    ).tolist()
                )
                action = torch.clamp(
                    torch.from_numpy(raw_action).to(device).float(),
                    action_low,
                    action_high,
                )
                action[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = student.step(action)
                history.append(action.detach().clone())
                previous_action = action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                countdown -= 1
                reward_np = reward.detach().cpu().numpy().reshape(-1)
                batch_final[active] = reward_np[active]
                batch_max[active] = np.maximum(
                    batch_max[active], reward_np[active]
                )
                if "success" in info:
                    success_once |= (
                        info["success"].detach().cpu().numpy().reshape(-1).astype(bool)
                    )
                done = (
                    torch.logical_or(terminated, truncated)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool)
                )
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(np.sum(newly_done)))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(np.sum(active)))
            successes.extend(success_once.astype(float).tolist())
            final_rewards.extend(batch_final.astype(float).tolist())
            max_rewards.extend(batch_max.astype(float).tolist())
        finally:
            student.close()
            if branch is not None:
                branch.close()
    progress.close()
    success = float(np.mean(successes))
    payload = {
        "stage": "closed_loop_interface",
        "candidate": candidate,
        "goal_source": goal_source,
        "seed": seed,
        "checkpoint": str(hierarchy_path),
        "episodes": eval_episodes,
        "seed_start": seed_start,
        "success": success,
        "success_wilson_95": _wilson_interval(success, eval_episodes),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "teacher_action_mae": float(np.mean(teacher_maes)),
        "high_level_decisions_per_episode": high_decisions / eval_episodes,
        "normalized_goal_prediction_l2": (
            float(np.mean(goal_errors)) if goal_errors else None
        ),
        "replay_current_state_error_max": (
            float(np.max(replay_errors)) if replay_errors else None
        ),
        "offline_validation": checkpoint["validation_metrics"],
        "representation_validation": representation_checkpoint[
            "validation_metrics"
        ],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def run_learned_interface_candidate(
    config: Config,
    candidate: str,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    representation = train_learned_interface_representation(
        config, candidate, seed, force=force
    )
    probe = probe_learned_interface_representation(
        config, candidate, seed, force=force
    )
    hierarchy = train_learned_interface_hierarchy(
        config, candidate, seed, force=force
    )
    learned = evaluate_learned_interface_hierarchy(
        config,
        candidate,
        "learned",
        seed,
        episodes=episodes,
        force=force,
    )
    oracle = evaluate_learned_interface_hierarchy(
        config,
        candidate,
        "oracle",
        seed,
        episodes=episodes,
        force=force,
    )
    return {
        "representation": representation,
        "probe": probe,
        "hierarchy": hierarchy,
        "learned": learned,
        "oracle": oracle,
    }
