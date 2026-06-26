from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import torch
from rich.console import Console

from hcl_poc.config import Config, load_config
from hcl_poc.data import prepare_dataset
from hcl_poc.eval import evaluate, horizon_steps, record_videos
from hcl_poc.goal_diagnostics import (
    aggregate_goal_diagnostics,
    learned_interface_goal_diagnostics,
)
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
    audit_learned_interface_reset_vectorization,
    compare_learned_interface_eval_jsons,
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
    compare_serial_low_level_eval,
    compare_serial_low_level_segments,
    evaluate_residual_rl,
    evaluate_residual_rl_serial,
    export_direct_low_as_hierarchy,
    fit_serial_initial_selector,
    fit_serial_segment_selector,
    record_low_level_rl_videos,
    train_direct_low_rl,
    train_residual_rl,
)
from hcl_poc.privileged_z import (
    collect_privileged_z_closed_loop_action_search_bank,
    collect_privileged_z_closed_loop_preserve_bank,
    create_privileged_z_hard_case_manifest,
    evaluate_privileged_z_local_action_search,
    evaluate_privileged_z_local_paired,
    evaluate_privileged_z_hierarchy,
    evaluate_privileged_z_goal_validity,
    evaluate_privileged_z_branch_outcomes,
    filter_privileged_z_action_search_bank,
    reweight_privileged_z_action_search_bank,
    train_privileged_z_local_replay_distill,
    train_privileged_z_direct_rl,
    train_privileged_z_hierarchy,
    train_privileged_z_residual_rl,
)
from hcl_poc.reachability import (
    evaluate_reachability_distance,
    train_reachability_distance,
)
from hcl_poc.report import build_report
from hcl_poc.rl import collect_ppo_dataset, evaluate_ppo, ppo_status, train_ppo
from hcl_poc.rl_rerun import (
    audit_rl_rerun_state_dataset,
    audit_rl_rerun_vector_dataset,
    audit_rl_rerun_local_mode_a,
    audit_rl_rerun_local_sample_proxies,
    compare_rl_rerun_local_proxy_audits,
    collect_rl_rerun_state_dataset,
    collect_rl_rerun_vector_dataset,
    create_rl_rerun_local_eval_manifest,
    ensure_rl_rerun_action_aliases,
    evaluate_rl_rerun_closed_loop_r1,
    evaluate_rl_rerun_closed_loop_r2,
    evaluate_rl_rerun_closed_loop_r3,
    evaluate_rl_rerun_learned_goal_validity,
    evaluate_rl_rerun_local_r1,
    evaluate_rl_rerun_local_r2,
    evaluate_rl_rerun_local_r3,
    fit_rl_rerun_closed_loop_selector,
    fit_rl_rerun_oracle_segment_selector,
    run_rl_rerun_algorithm_audit,
    run_rl_rerun_local_reset_audit,
    run_rl_rerun_throughput_benchmark,
    record_rl_rerun_videos,
    train_rl_rerun_low_flow_base,
    train_rl_rerun_local_r1,
    train_rl_rerun_local_r2,
    train_rl_rerun_local_r3,
    train_rl_rerun_supervised_point,
)
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
    vae_scaling_config,
    validate_nested_vae_scaling_manifests,
)

console = Console()


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/pusht.yaml")


def _low_level_config_with_overrides(config: Config, args: argparse.Namespace) -> Config:
    raw = copy.deepcopy(config.raw)
    low = raw.setdefault("low_level_rl", {})
    for arg_name, config_name in [
        ("num_envs", "num_envs"),
        ("rollout_steps", "rollout_steps"),
        ("num_minibatches", "num_minibatches"),
        ("update_epochs", "update_epochs"),
        ("residual_penalty_weight", "residual_penalty_weight"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            low[config_name] = value
    learning_rate = getattr(args, "learning_rate", None)
    if learning_rate is not None:
        key = (
            "direct_learning_rate"
            if getattr(args, "low_level_rl_command", "") == "train-r3"
            else "learning_rate"
        )
        low[key] = learning_rate
    initial_logstd = getattr(args, "initial_logstd", None)
    if initial_logstd is not None:
        key = (
            "direct_initial_logstd"
            if getattr(args, "low_level_rl_command", "") == "train-r3"
            else "initial_logstd"
        )
        low[key] = initial_logstd
    if getattr(args, "no_segment_terminate_gae", False):
        low["segment_terminates_gae"] = False
    return type(config)(raw=raw, path=config.path)


def low_level_rl_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.low_level_rl_command == "audit":
        console.print(audit_low_level_rl(config, args.n_demo, args.seed))
    elif args.low_level_rl_command == "compare-serial":
        console.print(
            compare_serial_low_level_eval(
                Path(args.base_json),
                Path(args.candidate_json),
                output=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "compare-serial-segments":
        console.print(
            compare_serial_low_level_segments(
                Path(args.base_json),
                Path(args.candidate_json),
                output=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "fit-serial-selector":
        console.print(
            fit_serial_initial_selector(
                Path(args.base_json),
                Path(args.candidate_json),
                Path(args.output),
                validation_base_json=Path(args.validation_base_json)
                if args.validation_base_json
                else None,
                validation_candidate_json=Path(args.validation_candidate_json)
                if args.validation_candidate_json
                else None,
                ridge=args.ridge,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "fit-serial-segment-selector":
        console.print(
            fit_serial_segment_selector(
                Path(args.base_json),
                Path(args.candidate_json),
                Path(args.output),
                validation_base_json=Path(args.validation_base_json)
                if args.validation_base_json
                else None,
                validation_candidate_json=Path(args.validation_candidate_json)
                if args.validation_candidate_json
                else None,
                extra_base_jsons=[Path(path) for path in args.extra_base_json],
                extra_candidate_jsons=[
                    Path(path) for path in args.extra_candidate_json
                ],
                ridge=args.ridge,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "export-direct-hierarchy":
        console.print(
            export_direct_low_as_hierarchy(
                config,
                n_demo=args.n_demo,
                seed=args.seed,
                candidate=args.candidate,
                checkpoint_path=Path(args.checkpoint),
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "train-r1":
        run_config = _low_level_config_with_overrides(config, args)
        console.print(
            train_residual_rl(
                run_config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                alpha=args.alpha,
                terminal_weight=args.terminal_weight,
                distance_progress_weight=args.distance_progress_weight,
                task_reward_weight=args.task_reward_weight,
                task_progress_weight=args.task_progress_weight,
                distance_metric=args.distance_metric,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
                candidate=args.candidate,
                rl_seed_offset=args.rl_seed_offset,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "train-r3":
        run_config = _low_level_config_with_overrides(config, args)
        console.print(
            train_direct_low_rl(
                run_config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                bc_weight=args.bc_weight,
                terminal_weight=args.terminal_weight,
                distance_progress_weight=args.distance_progress_weight,
                task_reward_weight=args.task_reward_weight,
                task_progress_weight=args.task_progress_weight,
                reward_mode=args.reward_mode,
                distance_metric=args.distance_metric,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
                candidate=args.candidate,
                rl_seed_offset=args.rl_seed_offset,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "eval-serial":
        console.print(
            evaluate_residual_rl_serial(
                config,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                episodes=args.episodes,
                seed_start=args.seed_start,
                candidate=args.candidate,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                distance_metric=args.distance_metric,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
                residual_l2_gate_max=args.residual_l2_gate_max,
                selected_distance_gate_max=args.selected_distance_gate_max,
                initial_selector_weights=args.initial_selector_weights,
                initial_selector_mean=args.initial_selector_mean,
                initial_selector_std=args.initial_selector_std,
                initial_selector_threshold=args.initial_selector_threshold,
                segment_selector_weights=args.segment_selector_weights,
                segment_selector_mean=args.segment_selector_mean,
                segment_selector_std=args.segment_selector_std,
                segment_selector_threshold=args.segment_selector_threshold,
                goal_source=args.goal_source,
                goal_projection=args.goal_projection,
                goal_projection_topk=args.goal_projection_topk,
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
                candidate=args.candidate,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                ensemble_checkpoint_paths=[Path(path) for path in args.ensemble_checkpoints]
                if args.ensemble_checkpoints
                else None,
                distance_metric=args.distance_metric,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
                residual_l2_gate_max=args.residual_l2_gate_max,
                selected_distance_gate_max=args.selected_distance_gate_max,
                initial_selector_weights=args.initial_selector_weights,
                initial_selector_mean=args.initial_selector_mean,
                initial_selector_std=args.initial_selector_std,
                initial_selector_threshold=args.initial_selector_threshold,
                force=args.force,
            )
        )
    elif args.low_level_rl_command == "video":
        for path in record_low_level_rl_videos(
            config,
            n_demo=args.n_demo,
            seed=args.seed,
            run_name=args.run_name,
            episodes=args.episodes,
            seed_start=args.seed_start,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            force=args.force,
        ):
            console.print(path)
    else:
        raise ValueError(args.low_level_rl_command)


def rl_rerun_cmd(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.rl_rerun_command == "collect-state-data":
        console.print(
            collect_rl_rerun_state_dataset(
                config,
                episodes=args.episodes,
                output_path=Path(args.output) if args.output else None,
                seed_start=args.seed_start,
                max_attempts=args.max_attempts,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                store_rgb=args.store_rgb,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "collect-vector-data":
        console.print(
            collect_rl_rerun_vector_dataset(
                config,
                output_path=Path(args.output) if args.output else None,
                num_envs=args.num_envs,
                batches=args.batches,
                max_steps=args.max_steps,
                seed_start=args.seed_start,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                store_dino=not args.no_store_dino,
                disturbed=args.disturbed,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "audit-state-data":
        console.print(
            audit_rl_rerun_state_dataset(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                samples=args.samples,
                horizon=args.horizon,
                seed=args.seed,
                recompute_dino=args.recompute_dino,
                warm_start_replay=args.warm_start_replay,
            )
        )
    elif args.rl_rerun_command == "audit-vector-data":
        console.print(
            audit_rl_rerun_vector_dataset(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                batches=args.batches,
                seed=args.seed,
                horizon=args.horizon,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "local-mode-a-audit":
        console.print(
            audit_rl_rerun_local_mode_a(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                manifest_path=Path(args.manifest) if args.manifest else None,
                output_path=Path(args.output) if args.output else None,
                include_samples=args.include_samples,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
            )
        )
    elif args.rl_rerun_command == "create-local-eval-manifest":
        console.print(
            create_rl_rerun_local_eval_manifest(
                dataset_path=Path(args.dataset),
                output_path=Path(args.output),
                episodes=args.episodes,
                seed=args.seed,
                horizon=args.horizon,
            )
        )
    elif args.rl_rerun_command == "train-local-r1":
        console.print(
            train_rl_rerun_local_r1(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                alpha=args.alpha,
                terminal_weight=args.terminal_weight,
                residual_penalty_weight=args.residual_penalty_weight,
                learning_rate=args.learning_rate,
                num_minibatches=args.num_minibatches,
                checkpoint_every_updates=args.checkpoint_every_updates,
                initial_logstd=args.initial_logstd,
                residual_condition_mode=args.residual_condition_mode,
                residual_action_mode=args.residual_action_mode,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-local-r1":
        console.print(
            evaluate_rl_rerun_local_r1(
                config,
                checkpoint_path=Path(args.checkpoint),
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                manifest_path=Path(args.manifest) if args.manifest else None,
                output_path=Path(args.output) if args.output else None,
                include_samples=args.include_samples,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
            )
        )
    elif args.rl_rerun_command == "eval-closed-loop-r1":
        console.print(
            evaluate_rl_rerun_closed_loop_r1(
                config,
                checkpoint_path=Path(args.checkpoint),
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                eval_seed_start=args.eval_seed_start,
                num_envs=args.num_envs,
                disturbed=args.disturbed,
                goal_source=args.goal_source,
                oracle_copy_mode=args.oracle_copy_mode,
                action_delta_gate_min=args.action_delta_gate_min,
                goal_l2_gate_min=args.goal_l2_gate_min,
                step_selector_path=Path(args.step_selector) if args.step_selector else None,
                segment_selector_path=Path(args.segment_selector)
                if args.segment_selector
                else None,
                oracle_segment_selector=args.oracle_segment_selector,
                oracle_segment_selector_metric=args.oracle_segment_selector_metric,
                diagnose_oracle_goals=args.diagnose_oracle_goals,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "train-low-flow-base":
        console.print(
            train_rl_rerun_low_flow_base(
                config,
                n_demo=args.n_demo,
                seed=args.seed,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-local-r2":
        console.print(
            train_rl_rerun_local_r2(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                alpha=args.alpha,
                terminal_weight=args.terminal_weight,
                residual_penalty_weight=args.residual_penalty_weight,
                learning_rate=args.learning_rate,
                num_minibatches=args.num_minibatches,
                checkpoint_every_updates=args.checkpoint_every_updates,
                initial_logstd=args.initial_logstd,
                flow_checkpoint_path=Path(args.flow_checkpoint)
                if args.flow_checkpoint
                else None,
                residual_condition_mode=args.residual_condition_mode,
                residual_action_mode=args.residual_action_mode,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-local-r2":
        console.print(
            evaluate_rl_rerun_local_r2(
                config,
                checkpoint_path=Path(args.checkpoint),
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                manifest_path=Path(args.manifest) if args.manifest else None,
                output_path=Path(args.output) if args.output else None,
                include_samples=args.include_samples,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
            )
        )
    elif args.rl_rerun_command == "eval-closed-loop-r2":
        console.print(
            evaluate_rl_rerun_closed_loop_r2(
                config,
                checkpoint_path=Path(args.checkpoint),
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                eval_seed_start=args.eval_seed_start,
                num_envs=args.num_envs,
                disturbed=args.disturbed,
                goal_source=args.goal_source,
                oracle_copy_mode=args.oracle_copy_mode,
                action_delta_gate_min=args.action_delta_gate_min,
                goal_l2_gate_min=args.goal_l2_gate_min,
                step_selector_path=Path(args.step_selector) if args.step_selector else None,
                segment_selector_path=Path(args.segment_selector)
                if args.segment_selector
                else None,
                oracle_segment_selector=args.oracle_segment_selector,
                oracle_segment_selector_metric=args.oracle_segment_selector_metric,
                diagnose_oracle_goals=args.diagnose_oracle_goals,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "train-local-r3":
        console.print(
            train_rl_rerun_local_r3(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                run_name=args.run_name,
                total_steps=args.steps,
                bc_weight=args.bc_weight,
                terminal_weight=args.terminal_weight,
                dense_progress_weight=args.dense_progress_weight,
                task_reward_weight=args.task_reward_weight,
                reward_mode=args.reward_mode,
                reward_distance_metric=args.reward_distance_metric,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
                learning_rate=args.learning_rate,
                num_minibatches=args.num_minibatches,
                initial_logstd=args.initial_logstd,
                checkpoint_every_updates=args.checkpoint_every_updates,
                goal_sensitivity_weight=args.goal_sensitivity_weight,
                goal_sensitivity_margin=args.goal_sensitivity_margin,
                min_base_terminal_distance=args.min_base_terminal_distance,
                max_base_terminal_env_reward=args.max_base_terminal_env_reward,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-local-r3":
        console.print(
            evaluate_rl_rerun_local_r3(
                config,
                checkpoint_path=Path(args.checkpoint),
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                manifest_path=Path(args.manifest) if args.manifest else None,
                output_path=Path(args.output) if args.output else None,
                include_samples=args.include_samples,
                reachability_checkpoint_path=Path(args.reachability_checkpoint)
                if args.reachability_checkpoint
                else None,
            )
        )
    elif args.rl_rerun_command == "eval-closed-loop-r3":
        console.print(
            evaluate_rl_rerun_closed_loop_r3(
                config,
                checkpoint_path=Path(args.checkpoint),
                n_demo=args.n_demo,
                seed=args.seed,
                episodes=args.episodes,
                eval_seed_start=args.eval_seed_start,
                num_envs=args.num_envs,
                disturbed=args.disturbed,
                goal_source=args.goal_source,
                oracle_copy_mode=args.oracle_copy_mode,
                action_delta_gate_min=args.action_delta_gate_min,
                goal_l2_gate_min=args.goal_l2_gate_min,
                step_selector_path=Path(args.step_selector) if args.step_selector else None,
                segment_selector_path=Path(args.segment_selector)
                if args.segment_selector
                else None,
                oracle_segment_selector=args.oracle_segment_selector,
                oracle_segment_selector_metric=args.oracle_segment_selector_metric,
                diagnose_oracle_goals=args.diagnose_oracle_goals,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "eval-learned-goal-validity":
        console.print(
            evaluate_rl_rerun_learned_goal_validity(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                samples=args.samples,
                sample_seed=args.sample_seed,
                horizon=args.horizon,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "fit-closed-loop-selector":
        console.print(
            fit_rl_rerun_closed_loop_selector(
                train_json_path=Path(args.train_json),
                validation_json_path=Path(args.validation_json)
                if args.validation_json
                else None,
                output_path=Path(args.output),
                feature_names=args.feature_names,
                ridge=args.ridge,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "fit-oracle-segment-selector":
        console.print(
            fit_rl_rerun_oracle_segment_selector(
                train_json_path=Path(args.train_json),
                validation_json_path=Path(args.validation_json)
                if args.validation_json
                else None,
                output_path=Path(args.output),
                feature_names=args.feature_names,
                ridge=args.ridge,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "audit-local-sample-proxies":
        console.print(
            audit_rl_rerun_local_sample_proxies(
                local_json_path=Path(args.local_json),
                output_path=Path(args.output),
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "compare-local-proxy-audits":
        console.print(
            compare_rl_rerun_local_proxy_audits(
                audit_paths=[Path(path) for path in args.audit_json],
                output_path=Path(args.output),
                names=args.name,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "record-videos":
        for path in record_rl_rerun_videos(
            config,
            checkpoint_path=Path(args.checkpoint),
            n_demo=args.n_demo,
            seed=args.seed,
            episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
            mode=args.mode,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            force=args.force,
        ):
            console.print(path)
    elif args.rl_rerun_command == "ensure-action-aliases":
        console.print(
            ensure_rl_rerun_action_aliases(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
            )
        )
    elif args.rl_rerun_command == "train-supervised":
        console.print(
            train_rl_rerun_supervised_point(
                config,
                n_demo=args.n_demo,
                seed=args.seed,
                dataset_path=Path(args.dataset) if args.dataset else None,
                eval_episodes=args.eval_episodes,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-privileged-z":
        console.print(
            train_privileged_z_hierarchy(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_trajectories=args.n_trajectories,
                validation_trajectories=args.validation_trajectories,
                horizon_steps=args.horizon,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                model_family=args.model_family,
                flow_steps=args.flow_steps,
                selection_mode=args.selection_mode,
                train_per_expert=args.train_per_expert,
                validation_per_expert=args.validation_per_expert,
                run_tag=args.run_tag,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-privileged-z":
        console.print(
            evaluate_privileged_z_hierarchy(
                config,
                checkpoint_path=Path(args.checkpoint),
                mode=args.mode,
                episodes=args.episodes,
                seed_start=args.seed_start,
                num_envs=args.num_envs,
                residual_checkpoint_path=Path(args.residual_checkpoint)
                if args.residual_checkpoint
                else None,
                tuned_gate_mode=args.tuned_gate_mode,
                tuned_gate_max_degradation_mse=args.tuned_gate_max_degradation_mse,
                high_goal_delta_scale=args.high_goal_delta_scale,
                high_goal_projection=args.high_goal_projection,
                high_goal_branch_bank_path=Path(args.high_goal_branch_bank)
                if args.high_goal_branch_bank
                else None,
                high_goal_branch_selector_path=Path(args.high_goal_branch_selector)
                if args.high_goal_branch_selector
                else None,
                high_goal_projection_state_weight=args.high_goal_projection_state_weight,
                high_goal_projection_goal_weight=args.high_goal_projection_goal_weight,
                high_goal_bank_episodes=args.high_goal_bank_episodes,
                high_goal_bank_seed_start=args.high_goal_bank_seed_start,
                high_goal_bank_num_envs=args.high_goal_bank_num_envs,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-privileged-z-local-paired":
        console.print(
            evaluate_privileged_z_local_paired(
                config,
                checkpoint_path=Path(args.checkpoint),
                manifest_path=Path(args.manifest),
                residual_checkpoint_path=Path(args.residual_checkpoint)
                if args.residual_checkpoint
                else None,
                output_path=Path(args.output) if args.output else None,
                goal_source=args.goal_source,
                success_epsilon=args.success_epsilon,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-privileged-z-goal-validity":
        console.print(
            evaluate_privileged_z_goal_validity(
                config,
                checkpoint_path=Path(args.checkpoint),
                episodes=args.episodes,
                seed_start=args.seed_start,
                num_envs=args.num_envs,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-privileged-z-local-action-search":
        console.print(
            evaluate_privileged_z_local_action_search(
                config,
                checkpoint_path=Path(args.checkpoint),
                manifest_path=Path(args.manifest),
                output_path=Path(args.output) if args.output else None,
                goal_source=args.goal_source,
                random_candidates=args.random_candidates,
                random_noise_std=args.random_noise_std,
                success_epsilon=args.success_epsilon,
                seed=args.seed,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "create-privileged-z-hard-case-manifest":
        console.print(
            create_privileged_z_hard_case_manifest(
                config,
                checkpoint_path=Path(args.checkpoint),
                manifest_path=Path(args.manifest),
                output_path=Path(args.output),
                goal_source=args.goal_source,
                threshold_mse=args.threshold_mse,
                max_envs_per_entry=args.max_envs_per_entry,
                seed=args.seed,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "filter-privileged-z-action-search-bank":
        console.print(
            filter_privileged_z_action_search_bank(
                Path(args.input),
                output_path=Path(args.output),
                min_base_mse=args.min_base_mse,
                max_base_mse=args.max_base_mse,
                min_best_mse=args.min_best_mse,
                max_best_mse=args.max_best_mse,
                min_improvement_mse=args.min_improvement_mse,
                max_improvement_mse=args.max_improvement_mse,
                max_action_delta_l2=args.max_action_delta_l2,
                max_oracle_delta_mse=args.max_oracle_delta_mse,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "reweight-privileged-z-action-search-bank":
        console.print(
            reweight_privileged_z_action_search_bank(
                Path(args.input),
                output_path=Path(args.output),
                mode=args.mode,
                success_epsilon=args.success_epsilon,
                improvement_scale=args.improvement_scale,
                min_weight=args.min_weight,
                max_weight=args.max_weight,
                normalize_mean=not args.no_normalize_mean,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-privileged-z-local-replay-distill":
        console.print(
            train_privileged_z_local_replay_distill(
                config,
                checkpoint_path=Path(args.checkpoint),
                manifest_path=Path(args.manifest),
                preserve_manifest_path=Path(args.preserve_manifest)
                if args.preserve_manifest
                else None,
                preserve_npz_path=Path(args.preserve_npz) if args.preserve_npz else None,
                improve_npz_path=Path(args.improve_npz) if args.improve_npz else None,
                replay_weight=args.replay_weight,
                preserve_weight=args.preserve_weight,
                preserve_npz_weight=args.preserve_npz_weight,
                improve_npz_weight=args.improve_npz_weight,
                run_tag=args.run_tag,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_scope=args.train_scope,
                initial_logstd=args.initial_logstd,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "collect-privileged-z-closed-loop-action-search-bank":
        console.print(
            collect_privileged_z_closed_loop_action_search_bank(
                config,
                checkpoint_path=Path(args.checkpoint),
                mode=args.mode,
                episodes=args.episodes,
                seed_start=args.seed_start,
                num_envs=args.num_envs,
                random_candidates=args.random_candidates,
                random_noise_std=args.random_noise_std,
                min_improvement_mse=args.min_improvement_mse,
                max_base_mse=args.max_base_mse,
                max_action_delta_l2=args.max_action_delta_l2,
                oracle_gate_max_degradation_mse=args.oracle_gate_max_degradation_mse,
                success_epsilon=args.success_epsilon,
                max_search_batches=args.max_search_batches,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "collect-privileged-z-closed-loop-preserve-bank":
        console.print(
            collect_privileged_z_closed_loop_preserve_bank(
                config,
                checkpoint_path=Path(args.checkpoint),
                mode=args.mode,
                episodes=args.episodes,
                seed_start=args.seed_start,
                num_envs=args.num_envs,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-privileged-z-branch-outcomes":
        console.print(
            evaluate_privileged_z_branch_outcomes(
                config,
                checkpoint_path=Path(args.checkpoint),
                episodes=args.episodes,
                seed_start=args.seed_start,
                num_envs=args.num_envs,
                random_candidates=args.random_candidates,
                random_noise_std=args.random_noise_std,
                branch_source=args.branch_source,
                branch_condition_goal_source=args.branch_condition_goal_source,
                min_improvement_mse=args.min_improvement_mse,
                max_action_delta_l2=args.max_action_delta_l2,
                max_branch_batches=args.max_branch_batches,
                max_rollout_steps=args.max_rollout_steps,
                bank_output_path=Path(args.bank_output) if args.bank_output else None,
                bank_min_success_delta=args.bank_min_success_delta,
                bank_min_return_delta=args.bank_min_return_delta,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-reachability-distance":
        run_config = (
            vae_scaling_config(config, args.n_demo)
            if args.n_demo is not None
            else config
        )
        console.print(
            train_reachability_distance(
                run_config,
                candidate=args.candidate,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                batches_per_epoch=args.batches_per_epoch,
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                lr=args.lr,
                horizon_steps=args.horizon_steps,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "eval-reachability-distance":
        run_config = (
            vae_scaling_config(config, args.n_demo)
            if args.n_demo is not None
            else config
        )
        console.print(
            evaluate_reachability_distance(
                run_config,
                candidate=args.candidate,
                seed=args.seed,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                samples=args.samples,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "goal-diagnostics":
        horizons = tuple(int(value) for value in args.horizons.split(",") if value)
        console.print(
            learned_interface_goal_diagnostics(
                config,
                n_demo=args.n_demo,
                candidate=args.candidate,
                seed=args.seed,
                samples=args.samples,
                horizons=horizons,
                output_path=Path(args.output) if args.output else None,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "aggregate-goal-diagnostics":
        console.print(
            aggregate_goal_diagnostics(
                args.input_glob,
                output_path=Path(args.output),
                min_goal_shuffle_l2=args.min_goal_shuffle_l2,
                min_goal_sensitivity_l2=args.min_goal_sensitivity_l2,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-privileged-z-residual":
        console.print(
            train_privileged_z_residual_rl(
                config,
                checkpoint_path=Path(args.checkpoint),
                init_dataset_path=Path(args.init_dataset),
                run_tag=args.run_tag,
                seed=args.seed,
                total_steps=args.steps,
                alpha=args.alpha,
                terminal_weight=args.terminal_weight,
                residual_penalty_weight=args.residual_penalty_weight,
                learning_rate=args.learning_rate,
                num_minibatches=args.num_minibatches,
                update_epochs=args.update_epochs,
                checkpoint_every_updates=args.checkpoint_every_updates,
                initial_logstd=args.initial_logstd,
                residual_action_mode=args.residual_action_mode,
                residual_goal_source=args.residual_goal_source,
                reward_mode=args.reward_mode,
                dense_progress_weight=args.dense_progress_weight,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "train-privileged-z-direct":
        console.print(
            train_privileged_z_direct_rl(
                config,
                checkpoint_path=Path(args.checkpoint),
                init_dataset_path=Path(args.init_dataset),
                run_tag=args.run_tag,
                direct_init_checkpoint_path=Path(args.direct_init_checkpoint)
                if args.direct_init_checkpoint
                else None,
                seed=args.seed,
                total_steps=args.steps,
                terminal_weight=args.terminal_weight,
                learning_rate=args.learning_rate,
                num_minibatches=args.num_minibatches,
                update_epochs=args.update_epochs,
                checkpoint_every_updates=args.checkpoint_every_updates,
                initial_logstd=args.initial_logstd,
                train_scope=args.train_scope,
                goal_source=args.goal_source,
                reward_mode=args.reward_mode,
                dense_progress_weight=args.dense_progress_weight,
                bc_weight=args.bc_weight,
                min_base_terminal_mse=args.min_base_terminal_mse,
                force=args.force,
            )
        )
    elif args.rl_rerun_command == "throughput-benchmark":
        console.print(
            run_rl_rerun_throughput_benchmark(
                config,
                num_envs_values=[
                    int(item.strip()) for item in args.num_envs.split(",") if item.strip()
                ],
                rollout_lens=[
                    int(item.strip()) for item in args.rollout_lens.split(",") if item.strip()
                ],
                n_demo=args.n_demo,
                seed=args.seed,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "algorithm-audit":
        console.print(
            run_rl_rerun_algorithm_audit(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                output_path=Path(args.output) if args.output else None,
            )
        )
    elif args.rl_rerun_command == "local-reset-audit":
        console.print(
            run_rl_rerun_local_reset_audit(
                config,
                dataset_path=Path(args.dataset) if args.dataset else None,
                n_demo=args.n_demo,
                seed=args.seed,
                num_envs=args.num_envs,
                batches=args.batches,
                output_path=Path(args.output) if args.output else None,
            )
        )
    else:
        raise ValueError(args.rl_rerun_command)


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
    if getattr(args, "seed", None) is not None or getattr(args, "rl_dir", None):
        raw = copy.deepcopy(config.raw)
        if getattr(args, "seed", None) is not None:
            raw["seed"] = int(args.seed)
        if getattr(args, "rl_dir", None):
            raw.setdefault("paths", {})["rl_dir"] = args.rl_dir
        config = type(config)(raw=raw, path=config.path)
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
            eval_seed_start=args.eval_seed_start,
            eval_num_envs=args.eval_num_envs,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-compare-evals":
        compare_learned_interface_eval_jsons(
            eval_jsons=[Path(path) for path in args.eval_json],
            names=args.name,
            output=Path(args.output),
            force=args.force,
        )
    elif args.incremental_command == "learned-interface-audit-reset-vectorization":
        audit_learned_interface_reset_vectorization(
            config,
            seed_start=args.seed_start,
            episodes=args.episodes,
            eval_num_envs=args.eval_num_envs,
            output=Path(args.output),
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
    compare_serial = low_level_rl_sub.add_parser("compare-serial")
    compare_serial.add_argument("--base-json", required=True)
    compare_serial.add_argument("--candidate-json", required=True)
    compare_serial.add_argument("--output")
    compare_serial.add_argument("--force", action="store_true")
    compare_serial.set_defaults(func=low_level_rl_cmd)
    compare_serial_segments = low_level_rl_sub.add_parser("compare-serial-segments")
    compare_serial_segments.add_argument("--base-json", required=True)
    compare_serial_segments.add_argument("--candidate-json", required=True)
    compare_serial_segments.add_argument("--output")
    compare_serial_segments.add_argument("--force", action="store_true")
    compare_serial_segments.set_defaults(func=low_level_rl_cmd)
    fit_serial_selector = low_level_rl_sub.add_parser("fit-serial-selector")
    fit_serial_selector.add_argument("--base-json", required=True)
    fit_serial_selector.add_argument("--candidate-json", required=True)
    fit_serial_selector.add_argument("--validation-base-json")
    fit_serial_selector.add_argument("--validation-candidate-json")
    fit_serial_selector.add_argument("--output", required=True)
    fit_serial_selector.add_argument("--ridge", type=float, default=1.0)
    fit_serial_selector.add_argument("--force", action="store_true")
    fit_serial_selector.set_defaults(func=low_level_rl_cmd)
    fit_serial_segment_selector_parser = low_level_rl_sub.add_parser(
        "fit-serial-segment-selector"
    )
    fit_serial_segment_selector_parser.add_argument("--base-json", required=True)
    fit_serial_segment_selector_parser.add_argument("--candidate-json", required=True)
    fit_serial_segment_selector_parser.add_argument(
        "--extra-base-json",
        action="append",
        default=[],
    )
    fit_serial_segment_selector_parser.add_argument(
        "--extra-candidate-json",
        action="append",
        default=[],
    )
    fit_serial_segment_selector_parser.add_argument("--validation-base-json")
    fit_serial_segment_selector_parser.add_argument("--validation-candidate-json")
    fit_serial_segment_selector_parser.add_argument("--output", required=True)
    fit_serial_segment_selector_parser.add_argument("--ridge", type=float, default=1.0)
    fit_serial_segment_selector_parser.add_argument("--force", action="store_true")
    fit_serial_segment_selector_parser.set_defaults(func=low_level_rl_cmd)
    export_direct_hierarchy = low_level_rl_sub.add_parser("export-direct-hierarchy")
    export_direct_hierarchy.add_argument(
        "--n-demo", type=int, choices=[500, 1000], required=True
    )
    export_direct_hierarchy.add_argument("--candidate", default="vae512_w2048_b1e6")
    export_direct_hierarchy.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    export_direct_hierarchy.add_argument("--checkpoint", required=True)
    export_direct_hierarchy.add_argument("--output")
    export_direct_hierarchy.add_argument("--force", action="store_true")
    export_direct_hierarchy.set_defaults(func=low_level_rl_cmd)
    train_r1 = low_level_rl_sub.add_parser("train-r1")
    train_r1.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    train_r1.add_argument("--candidate", default="vae512_w2048_b1e6")
    train_r1.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    train_r1.add_argument("--run-name", required=True)
    train_r1.add_argument("--steps", type=int, required=True)
    train_r1.add_argument("--alpha", type=float, default=0.1)
    train_r1.add_argument("--terminal-weight", type=float, default=1.0)
    train_r1.add_argument("--distance-progress-weight", type=float, default=1.0)
    train_r1.add_argument("--task-reward-weight", type=float, default=0.0)
    train_r1.add_argument("--task-progress-weight", type=float, default=0.0)
    train_r1.add_argument(
        "--distance-metric",
        choices=["raw_l2", "reachability"],
        default="raw_l2",
    )
    train_r1.add_argument("--reachability-checkpoint")
    train_r1.add_argument("--num-envs", type=int)
    train_r1.add_argument("--rollout-steps", type=int)
    train_r1.add_argument("--num-minibatches", type=int)
    train_r1.add_argument("--update-epochs", type=int)
    train_r1.add_argument("--learning-rate", type=float)
    train_r1.add_argument("--initial-logstd", type=float)
    train_r1.add_argument("--residual-penalty-weight", type=float)
    train_r1.add_argument("--rl-seed-offset", type=int, default=0)
    train_r1.add_argument("--no-segment-terminate-gae", action="store_true")
    train_r1.add_argument("--force", action="store_true")
    train_r1.set_defaults(func=low_level_rl_cmd)
    train_r3 = low_level_rl_sub.add_parser("train-r3")
    train_r3.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    train_r3.add_argument("--candidate", default="vae512_w2048_b1e6")
    train_r3.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    train_r3.add_argument("--run-name", required=True)
    train_r3.add_argument("--steps", type=int, required=True)
    train_r3.add_argument("--bc-weight", type=float, default=1.0)
    train_r3.add_argument("--terminal-weight", type=float, default=1.0)
    train_r3.add_argument("--distance-progress-weight", type=float, default=1.0)
    train_r3.add_argument("--task-reward-weight", type=float, default=0.0)
    train_r3.add_argument("--task-progress-weight", type=float, default=0.0)
    train_r3.add_argument(
        "--reward-mode",
        choices=["absolute", "paired"],
        default="absolute",
    )
    train_r3.add_argument(
        "--distance-metric",
        choices=["raw_l2", "reachability"],
        default="raw_l2",
    )
    train_r3.add_argument("--reachability-checkpoint")
    train_r3.add_argument("--num-envs", type=int)
    train_r3.add_argument("--rollout-steps", type=int)
    train_r3.add_argument("--num-minibatches", type=int)
    train_r3.add_argument("--update-epochs", type=int)
    train_r3.add_argument("--learning-rate", type=float)
    train_r3.add_argument("--initial-logstd", type=float)
    train_r3.add_argument("--residual-penalty-weight", type=float)
    train_r3.add_argument("--rl-seed-offset", type=int, default=0)
    train_r3.add_argument("--no-segment-terminate-gae", action="store_true")
    train_r3.add_argument("--force", action="store_true")
    train_r3.set_defaults(func=low_level_rl_cmd)
    low_eval = low_level_rl_sub.add_parser("eval")
    low_eval.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    low_eval.add_argument("--candidate", default="vae512_w2048_b1e6")
    low_eval.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    low_eval.add_argument("--run-name", required=True)
    low_eval.add_argument("--episodes", type=int, required=True)
    low_eval.add_argument("--seed-start", type=int, required=True)
    low_eval.add_argument("--checkpoint")
    low_eval.add_argument("--ensemble-checkpoints", nargs="+")
    low_eval.add_argument("--residual-l2-gate-max", type=float)
    low_eval.add_argument("--selected-distance-gate-max", type=float)
    low_eval.add_argument("--initial-selector-weights", nargs=3, type=float)
    low_eval.add_argument("--initial-selector-mean", nargs=3, type=float)
    low_eval.add_argument("--initial-selector-std", nargs=3, type=float)
    low_eval.add_argument("--initial-selector-threshold", type=float)
    low_eval.add_argument(
        "--distance-metric",
        choices=["raw_l2", "reachability"],
        default="raw_l2",
    )
    low_eval.add_argument("--reachability-checkpoint")
    low_eval.add_argument("--force", action="store_true")
    low_eval.set_defaults(func=low_level_rl_cmd)
    low_eval_serial = low_level_rl_sub.add_parser("eval-serial")
    low_eval_serial.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    low_eval_serial.add_argument("--candidate", default="vae512_w2048_b1e6")
    low_eval_serial.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    low_eval_serial.add_argument("--run-name", required=True)
    low_eval_serial.add_argument("--episodes", type=int, required=True)
    low_eval_serial.add_argument("--seed-start", type=int, required=True)
    low_eval_serial.add_argument("--checkpoint")
    low_eval_serial.add_argument("--residual-l2-gate-max", type=float)
    low_eval_serial.add_argument("--selected-distance-gate-max", type=float)
    low_eval_serial.add_argument("--initial-selector-weights", nargs=3, type=float)
    low_eval_serial.add_argument("--initial-selector-mean", nargs=3, type=float)
    low_eval_serial.add_argument("--initial-selector-std", nargs=3, type=float)
    low_eval_serial.add_argument("--initial-selector-threshold", type=float)
    low_eval_serial.add_argument("--segment-selector-weights", nargs=5, type=float)
    low_eval_serial.add_argument("--segment-selector-mean", nargs=5, type=float)
    low_eval_serial.add_argument("--segment-selector-std", nargs=5, type=float)
    low_eval_serial.add_argument("--segment-selector-threshold", type=float)
    low_eval_serial.add_argument(
        "--goal-source",
        choices=["learned", "oracle"],
        default="learned",
    )
    low_eval_serial.add_argument(
        "--goal-projection",
        choices=["none", "nearest_train", "nearest_train_dphi"],
        default="none",
    )
    low_eval_serial.add_argument("--goal-projection-topk", type=int, default=32)
    low_eval_serial.add_argument(
        "--distance-metric",
        choices=["raw_l2", "reachability"],
        default="raw_l2",
    )
    low_eval_serial.add_argument("--reachability-checkpoint")
    low_eval_serial.add_argument("--force", action="store_true")
    low_eval_serial.set_defaults(func=low_level_rl_cmd)
    low_video = low_level_rl_sub.add_parser("video")
    low_video.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    low_video.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    low_video.add_argument("--run-name", required=True)
    low_video.add_argument("--episodes", type=int, default=2)
    low_video.add_argument("--seed-start", type=int, required=True)
    low_video.add_argument("--checkpoint")
    low_video.add_argument("--force", action="store_true")
    low_video.set_defaults(func=low_level_rl_cmd)

    p = sub.add_parser("rl-rerun")
    add_config_arg(p)
    rl_rerun_sub = p.add_subparsers(dest="rl_rerun_command", required=True)
    collect_state = rl_rerun_sub.add_parser("collect-state-data")
    collect_state.add_argument("--episodes", type=int, required=True)
    collect_state.add_argument("--output")
    collect_state.add_argument("--seed-start", type=int)
    collect_state.add_argument("--max-attempts", type=int)
    collect_state.add_argument("--checkpoint")
    collect_state.add_argument("--store-rgb", action="store_true")
    collect_state.add_argument("--force", action="store_true")
    collect_state.set_defaults(func=rl_rerun_cmd)
    collect_vector = rl_rerun_sub.add_parser("collect-vector-data")
    collect_vector.add_argument("--output")
    collect_vector.add_argument("--num-envs", type=int, default=16)
    collect_vector.add_argument("--batches", type=int, default=2)
    collect_vector.add_argument("--max-steps", type=int, default=60)
    collect_vector.add_argument("--seed-start", type=int, default=9_500_000)
    collect_vector.add_argument("--checkpoint")
    collect_vector.add_argument("--no-store-dino", action="store_true")
    collect_vector.add_argument("--disturbed", action="store_true")
    collect_vector.add_argument("--force", action="store_true")
    collect_vector.set_defaults(func=rl_rerun_cmd)
    audit_state = rl_rerun_sub.add_parser("audit-state-data")
    audit_state.add_argument("--dataset")
    audit_state.add_argument("--samples", type=int, default=100)
    audit_state.add_argument("--horizon", type=int, default=10)
    audit_state.add_argument("--seed", type=int, default=0)
    audit_state.add_argument("--recompute-dino", action="store_true")
    audit_state.add_argument("--warm-start-replay", action="store_true")
    audit_state.set_defaults(func=rl_rerun_cmd)
    audit_vector = rl_rerun_sub.add_parser("audit-vector-data")
    audit_vector.add_argument("--dataset")
    audit_vector.add_argument("--batches", type=int, default=4)
    audit_vector.add_argument("--seed", type=int, default=0)
    audit_vector.add_argument("--horizon", type=int, default=10)
    audit_vector.add_argument("--output")
    audit_vector.set_defaults(func=rl_rerun_cmd)
    local_mode_a = rl_rerun_sub.add_parser("local-mode-a-audit")
    local_mode_a.add_argument("--dataset")
    local_mode_a.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    local_mode_a.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    local_mode_a.add_argument("--episodes", type=int, default=4)
    local_mode_a.add_argument("--manifest")
    local_mode_a.add_argument("--output")
    local_mode_a.set_defaults(func=rl_rerun_cmd)
    local_manifest = rl_rerun_sub.add_parser("create-local-eval-manifest")
    local_manifest.add_argument("--dataset", required=True)
    local_manifest.add_argument("--output", required=True)
    local_manifest.add_argument("--episodes", type=int, required=True)
    local_manifest.add_argument("--seed", type=int, default=0)
    local_manifest.add_argument("--horizon", type=int, default=10)
    local_manifest.set_defaults(func=rl_rerun_cmd)
    train_local_r1 = rl_rerun_sub.add_parser("train-local-r1")
    train_local_r1.add_argument("--dataset")
    train_local_r1.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    train_local_r1.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    train_local_r1.add_argument("--run-name", default="local_r1_mode_a")
    train_local_r1.add_argument("--steps", type=int, default=32768)
    train_local_r1.add_argument("--alpha", type=float, default=0.1)
    train_local_r1.add_argument("--terminal-weight", type=float, default=1.0)
    train_local_r1.add_argument("--residual-penalty-weight", type=float)
    train_local_r1.add_argument("--learning-rate", type=float)
    train_local_r1.add_argument("--num-minibatches", type=int)
    train_local_r1.add_argument("--checkpoint-every-updates", type=int, default=5)
    train_local_r1.add_argument("--initial-logstd", type=float)
    train_local_r1.add_argument(
        "--residual-condition-mode",
        choices=["full", "goal_delta"],
        default="full",
    )
    train_local_r1.add_argument(
        "--residual-action-mode",
        choices=["additive", "margin_scaled"],
        default="additive",
    )
    train_local_r1.add_argument("--force", action="store_true")
    train_local_r1.set_defaults(func=rl_rerun_cmd)
    eval_local_r1 = rl_rerun_sub.add_parser("eval-local-r1")
    eval_local_r1.add_argument("--checkpoint", required=True)
    eval_local_r1.add_argument("--dataset")
    eval_local_r1.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    eval_local_r1.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_local_r1.add_argument("--episodes", type=int, default=4)
    eval_local_r1.add_argument("--manifest")
    eval_local_r1.add_argument("--include-samples", action="store_true")
    eval_local_r1.add_argument("--reachability-checkpoint")
    eval_local_r1.add_argument("--output")
    eval_local_r1.set_defaults(func=rl_rerun_cmd)
    eval_closed_loop_r1 = rl_rerun_sub.add_parser("eval-closed-loop-r1")
    eval_closed_loop_r1.add_argument("--checkpoint", required=True)
    eval_closed_loop_r1.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    eval_closed_loop_r1.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_closed_loop_r1.add_argument("--episodes", type=int, default=100)
    eval_closed_loop_r1.add_argument("--eval-seed-start", type=int, default=10_000)
    eval_closed_loop_r1.add_argument("--num-envs", type=int, default=64)
    eval_closed_loop_r1.add_argument("--disturbed", action="store_true")
    eval_closed_loop_r1.add_argument(
        "--goal-source", choices=["learned", "oracle"], default="learned"
    )
    eval_closed_loop_r1.add_argument(
        "--oracle-copy-mode", choices=["replay", "state_dict"], default="replay"
    )
    eval_closed_loop_r1.add_argument("--action-delta-gate-min", type=float)
    eval_closed_loop_r1.add_argument("--goal-l2-gate-min", type=float)
    eval_closed_loop_r1.add_argument("--step-selector")
    eval_closed_loop_r1.add_argument("--segment-selector")
    eval_closed_loop_r1.add_argument("--oracle-segment-selector", action="store_true")
    eval_closed_loop_r1.add_argument(
        "--oracle-segment-selector-metric",
        choices=["latent_distance", "env_reward", "env_max_reward", "success"],
        default="latent_distance",
    )
    eval_closed_loop_r1.add_argument("--diagnose-oracle-goals", action="store_true")
    eval_closed_loop_r1.add_argument("--output")
    eval_closed_loop_r1.set_defaults(func=rl_rerun_cmd)
    low_flow_base = rl_rerun_sub.add_parser("train-low-flow-base")
    low_flow_base.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    low_flow_base.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    low_flow_base.add_argument("--force", action="store_true")
    low_flow_base.set_defaults(func=rl_rerun_cmd)
    train_local_r2 = rl_rerun_sub.add_parser("train-local-r2")
    train_local_r2.add_argument("--dataset")
    train_local_r2.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    train_local_r2.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    train_local_r2.add_argument("--run-name", default="local_r2_flow_residual")
    train_local_r2.add_argument("--steps", type=int, default=32768)
    train_local_r2.add_argument("--alpha", type=float, default=0.1)
    train_local_r2.add_argument("--terminal-weight", type=float, default=1.0)
    train_local_r2.add_argument("--residual-penalty-weight", type=float)
    train_local_r2.add_argument("--learning-rate", type=float)
    train_local_r2.add_argument("--num-minibatches", type=int)
    train_local_r2.add_argument("--checkpoint-every-updates", type=int, default=5)
    train_local_r2.add_argument("--initial-logstd", type=float)
    train_local_r2.add_argument("--flow-checkpoint")
    train_local_r2.add_argument(
        "--residual-condition-mode",
        choices=["full", "goal_delta"],
        default="full",
    )
    train_local_r2.add_argument(
        "--residual-action-mode",
        choices=["additive", "margin_scaled"],
        default="additive",
    )
    train_local_r2.add_argument("--force", action="store_true")
    train_local_r2.set_defaults(func=rl_rerun_cmd)
    eval_local_r2 = rl_rerun_sub.add_parser("eval-local-r2")
    eval_local_r2.add_argument("--checkpoint", required=True)
    eval_local_r2.add_argument("--dataset")
    eval_local_r2.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    eval_local_r2.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_local_r2.add_argument("--episodes", type=int, default=4)
    eval_local_r2.add_argument("--manifest")
    eval_local_r2.add_argument("--include-samples", action="store_true")
    eval_local_r2.add_argument("--reachability-checkpoint")
    eval_local_r2.add_argument("--output")
    eval_local_r2.set_defaults(func=rl_rerun_cmd)
    eval_closed_loop_r2 = rl_rerun_sub.add_parser("eval-closed-loop-r2")
    eval_closed_loop_r2.add_argument("--checkpoint", required=True)
    eval_closed_loop_r2.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    eval_closed_loop_r2.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_closed_loop_r2.add_argument("--episodes", type=int, default=100)
    eval_closed_loop_r2.add_argument("--eval-seed-start", type=int, default=10_000)
    eval_closed_loop_r2.add_argument("--num-envs", type=int, default=64)
    eval_closed_loop_r2.add_argument("--disturbed", action="store_true")
    eval_closed_loop_r2.add_argument(
        "--goal-source", choices=["learned", "oracle"], default="learned"
    )
    eval_closed_loop_r2.add_argument(
        "--oracle-copy-mode", choices=["replay", "state_dict"], default="replay"
    )
    eval_closed_loop_r2.add_argument("--action-delta-gate-min", type=float)
    eval_closed_loop_r2.add_argument("--goal-l2-gate-min", type=float)
    eval_closed_loop_r2.add_argument("--step-selector")
    eval_closed_loop_r2.add_argument("--segment-selector")
    eval_closed_loop_r2.add_argument("--oracle-segment-selector", action="store_true")
    eval_closed_loop_r2.add_argument(
        "--oracle-segment-selector-metric",
        choices=["latent_distance", "env_reward", "env_max_reward", "success"],
        default="latent_distance",
    )
    eval_closed_loop_r2.add_argument("--diagnose-oracle-goals", action="store_true")
    eval_closed_loop_r2.add_argument("--output")
    eval_closed_loop_r2.set_defaults(func=rl_rerun_cmd)
    train_local_r3 = rl_rerun_sub.add_parser("train-local-r3")
    train_local_r3.add_argument("--dataset")
    train_local_r3.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    train_local_r3.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    train_local_r3.add_argument("--run-name", default="local_r3_direct_last_layer")
    train_local_r3.add_argument("--steps", type=int, default=32768)
    train_local_r3.add_argument("--bc-weight", type=float, default=1.0)
    train_local_r3.add_argument("--terminal-weight", type=float, default=1.0)
    train_local_r3.add_argument("--dense-progress-weight", type=float, default=1.0)
    train_local_r3.add_argument("--task-reward-weight", type=float, default=0.0)
    train_local_r3.add_argument(
        "--reward-mode",
        choices=["progress", "paired", "task_paired"],
        default="progress",
    )
    train_local_r3.add_argument(
        "--reward-distance-metric",
        choices=["raw_l2", "reachability"],
        default="raw_l2",
    )
    train_local_r3.add_argument("--reachability-checkpoint")
    train_local_r3.add_argument("--learning-rate", type=float)
    train_local_r3.add_argument("--num-minibatches", type=int)
    train_local_r3.add_argument("--initial-logstd", type=float)
    train_local_r3.add_argument("--checkpoint-every-updates", type=int, default=5)
    train_local_r3.add_argument("--goal-sensitivity-weight", type=float, default=0.0)
    train_local_r3.add_argument("--goal-sensitivity-margin", type=float, default=0.05)
    train_local_r3.add_argument("--min-base-terminal-distance", type=float)
    train_local_r3.add_argument("--max-base-terminal-env-reward", type=float)
    train_local_r3.add_argument("--force", action="store_true")
    train_local_r3.set_defaults(func=rl_rerun_cmd)
    eval_local_r3 = rl_rerun_sub.add_parser("eval-local-r3")
    eval_local_r3.add_argument("--checkpoint", required=True)
    eval_local_r3.add_argument("--dataset")
    eval_local_r3.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    eval_local_r3.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_local_r3.add_argument("--episodes", type=int, default=4)
    eval_local_r3.add_argument("--manifest")
    eval_local_r3.add_argument("--include-samples", action="store_true")
    eval_local_r3.add_argument("--reachability-checkpoint")
    eval_local_r3.add_argument("--output")
    eval_local_r3.set_defaults(func=rl_rerun_cmd)
    eval_closed_loop_r3 = rl_rerun_sub.add_parser("eval-closed-loop-r3")
    eval_closed_loop_r3.add_argument("--checkpoint", required=True)
    eval_closed_loop_r3.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    eval_closed_loop_r3.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    eval_closed_loop_r3.add_argument("--episodes", type=int, default=100)
    eval_closed_loop_r3.add_argument("--eval-seed-start", type=int, default=10_000)
    eval_closed_loop_r3.add_argument("--num-envs", type=int, default=64)
    eval_closed_loop_r3.add_argument("--disturbed", action="store_true")
    eval_closed_loop_r3.add_argument(
        "--goal-source", choices=["learned", "oracle"], default="learned"
    )
    eval_closed_loop_r3.add_argument(
        "--oracle-copy-mode", choices=["replay", "state_dict"], default="replay"
    )
    eval_closed_loop_r3.add_argument("--action-delta-gate-min", type=float)
    eval_closed_loop_r3.add_argument("--goal-l2-gate-min", type=float)
    eval_closed_loop_r3.add_argument("--step-selector")
    eval_closed_loop_r3.add_argument("--segment-selector")
    eval_closed_loop_r3.add_argument("--oracle-segment-selector", action="store_true")
    eval_closed_loop_r3.add_argument(
        "--oracle-segment-selector-metric",
        choices=["latent_distance", "env_reward", "env_max_reward", "success"],
        default="latent_distance",
    )
    eval_closed_loop_r3.add_argument("--diagnose-oracle-goals", action="store_true")
    eval_closed_loop_r3.add_argument("--output")
    eval_closed_loop_r3.set_defaults(func=rl_rerun_cmd)
    eval_learned_goal_validity = rl_rerun_sub.add_parser(
        "eval-learned-goal-validity"
    )
    eval_learned_goal_validity.add_argument("--dataset")
    eval_learned_goal_validity.add_argument(
        "--n-demo", type=int, choices=[500, 1000], default=500
    )
    eval_learned_goal_validity.add_argument(
        "--seed", type=int, choices=[0, 1, 2], default=0
    )
    eval_learned_goal_validity.add_argument("--samples", type=int, default=4096)
    eval_learned_goal_validity.add_argument("--sample-seed", type=int, default=0)
    eval_learned_goal_validity.add_argument("--horizon", type=int)
    eval_learned_goal_validity.add_argument("--output")
    eval_learned_goal_validity.set_defaults(func=rl_rerun_cmd)
    fit_closed_loop_selector = rl_rerun_sub.add_parser("fit-closed-loop-selector")
    fit_closed_loop_selector.add_argument("--train-json", required=True)
    fit_closed_loop_selector.add_argument("--validation-json")
    fit_closed_loop_selector.add_argument("--output", required=True)
    fit_closed_loop_selector.add_argument("--feature-names", nargs="+")
    fit_closed_loop_selector.add_argument("--ridge", type=float, default=1.0)
    fit_closed_loop_selector.add_argument("--force", action="store_true")
    fit_closed_loop_selector.set_defaults(func=rl_rerun_cmd)
    fit_oracle_segment_selector = rl_rerun_sub.add_parser(
        "fit-oracle-segment-selector"
    )
    fit_oracle_segment_selector.add_argument("--train-json", required=True)
    fit_oracle_segment_selector.add_argument("--validation-json")
    fit_oracle_segment_selector.add_argument("--output", required=True)
    fit_oracle_segment_selector.add_argument("--feature-names", nargs="+")
    fit_oracle_segment_selector.add_argument("--ridge", type=float, default=1.0)
    fit_oracle_segment_selector.add_argument("--force", action="store_true")
    fit_oracle_segment_selector.set_defaults(func=rl_rerun_cmd)
    audit_local_sample_proxies = rl_rerun_sub.add_parser(
        "audit-local-sample-proxies"
    )
    audit_local_sample_proxies.add_argument("--local-json", required=True)
    audit_local_sample_proxies.add_argument("--output", required=True)
    audit_local_sample_proxies.add_argument("--force", action="store_true")
    audit_local_sample_proxies.set_defaults(func=rl_rerun_cmd)
    compare_local_proxy_audits = rl_rerun_sub.add_parser(
        "compare-local-proxy-audits"
    )
    compare_local_proxy_audits.add_argument("--audit-json", nargs="+", required=True)
    compare_local_proxy_audits.add_argument("--name", nargs="+")
    compare_local_proxy_audits.add_argument("--output", required=True)
    compare_local_proxy_audits.add_argument("--force", action="store_true")
    compare_local_proxy_audits.set_defaults(func=rl_rerun_cmd)
    record_rerun_videos = rl_rerun_sub.add_parser("record-videos")
    record_rerun_videos.add_argument("--checkpoint", required=True)
    record_rerun_videos.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    record_rerun_videos.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    record_rerun_videos.add_argument("--episodes", type=int, default=6)
    record_rerun_videos.add_argument("--eval-seed-start", type=int, default=10_000)
    record_rerun_videos.add_argument("--mode", choices=["frozen", "tuned", "both"], default="both")
    record_rerun_videos.add_argument("--output-dir")
    record_rerun_videos.add_argument("--force", action="store_true")
    record_rerun_videos.set_defaults(func=rl_rerun_cmd)
    aliases = rl_rerun_sub.add_parser("ensure-action-aliases")
    aliases.add_argument("--dataset")
    aliases.set_defaults(func=rl_rerun_cmd)
    train_supervised = rl_rerun_sub.add_parser("train-supervised")
    train_supervised.add_argument("--n-demo", type=int, choices=[500, 1000], required=True)
    train_supervised.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    train_supervised.add_argument("--dataset")
    train_supervised.add_argument("--eval-episodes", type=int, default=100)
    train_supervised.add_argument("--force", action="store_true")
    train_supervised.set_defaults(func=rl_rerun_cmd)
    train_priv_z = rl_rerun_sub.add_parser("train-privileged-z")
    train_priv_z.add_argument("--dataset")
    train_priv_z.add_argument("--n-trajectories", type=int, default=500)
    train_priv_z.add_argument("--validation-trajectories", type=int, default=200)
    train_priv_z.add_argument("--horizon", type=int, default=10)
    train_priv_z.add_argument("--seed", type=int, default=0)
    train_priv_z.add_argument("--epochs", type=int, default=40)
    train_priv_z.add_argument("--batch-size", type=int, default=4096)
    train_priv_z.add_argument("--hidden-dim", type=int, default=512)
    train_priv_z.add_argument("--lr", type=float, default=3e-4)
    train_priv_z.add_argument("--model-family", choices=["mlp", "flow"], default="mlp")
    train_priv_z.add_argument("--flow-steps", type=int, default=24)
    train_priv_z.add_argument(
        "--selection-mode",
        choices=["any_success", "balanced_experts"],
        default="any_success",
    )
    train_priv_z.add_argument("--train-per-expert", type=int)
    train_priv_z.add_argument("--validation-per-expert", type=int)
    train_priv_z.add_argument("--run-tag")
    train_priv_z.add_argument("--force", action="store_true")
    train_priv_z.set_defaults(func=rl_rerun_cmd)
    train_priv_z_residual = rl_rerun_sub.add_parser("train-privileged-z-residual")
    train_priv_z_residual.add_argument("--checkpoint", required=True)
    train_priv_z_residual.add_argument("--init-dataset", required=True)
    train_priv_z_residual.add_argument("--run-tag", required=True)
    train_priv_z_residual.add_argument("--seed", type=int, default=0)
    train_priv_z_residual.add_argument("--steps", type=int, default=32_768)
    train_priv_z_residual.add_argument("--alpha", type=float, default=0.1)
    train_priv_z_residual.add_argument("--terminal-weight", type=float, default=1.0)
    train_priv_z_residual.add_argument("--residual-penalty-weight", type=float, default=0.01)
    train_priv_z_residual.add_argument("--learning-rate", type=float, default=1e-4)
    train_priv_z_residual.add_argument("--num-minibatches", type=int, default=8)
    train_priv_z_residual.add_argument("--update-epochs", type=int, default=4)
    train_priv_z_residual.add_argument("--checkpoint-every-updates", type=int, default=5)
    train_priv_z_residual.add_argument("--initial-logstd", type=float, default=-2.3)
    train_priv_z_residual.add_argument(
        "--residual-action-mode",
        choices=["additive", "margin_scaled"],
        default="additive",
    )
    train_priv_z_residual.add_argument(
        "--residual-goal-source",
        choices=["oracle", "predicted", "oracle_to_predicted"],
        default="oracle",
    )
    train_priv_z_residual.add_argument(
        "--reward-mode",
        choices=["progress", "paired"],
        default="progress",
    )
    train_priv_z_residual.add_argument("--dense-progress-weight", type=float, default=0.0)
    train_priv_z_residual.add_argument("--force", action="store_true")
    train_priv_z_residual.set_defaults(func=rl_rerun_cmd)
    train_priv_z_direct = rl_rerun_sub.add_parser("train-privileged-z-direct")
    train_priv_z_direct.add_argument("--checkpoint", required=True)
    train_priv_z_direct.add_argument("--init-dataset", required=True)
    train_priv_z_direct.add_argument("--run-tag", required=True)
    train_priv_z_direct.add_argument("--direct-init-checkpoint")
    train_priv_z_direct.add_argument("--seed", type=int, default=0)
    train_priv_z_direct.add_argument("--steps", type=int, default=32_768)
    train_priv_z_direct.add_argument("--terminal-weight", type=float, default=1.0)
    train_priv_z_direct.add_argument("--learning-rate", type=float, default=3e-5)
    train_priv_z_direct.add_argument("--num-minibatches", type=int, default=8)
    train_priv_z_direct.add_argument("--update-epochs", type=int, default=4)
    train_priv_z_direct.add_argument("--checkpoint-every-updates", type=int, default=5)
    train_priv_z_direct.add_argument("--initial-logstd", type=float, default=-4.0)
    train_priv_z_direct.add_argument(
        "--train-scope",
        choices=["final_layer", "all"],
        default="final_layer",
    )
    train_priv_z_direct.add_argument(
        "--goal-source",
        choices=["oracle", "predicted", "oracle_to_predicted"],
        default="oracle",
    )
    train_priv_z_direct.add_argument(
        "--reward-mode",
        choices=["progress", "paired"],
        default="paired",
    )
    train_priv_z_direct.add_argument("--dense-progress-weight", type=float, default=0.0)
    train_priv_z_direct.add_argument("--bc-weight", type=float, default=0.0)
    train_priv_z_direct.add_argument("--min-base-terminal-mse", type=float)
    train_priv_z_direct.add_argument("--force", action="store_true")
    train_priv_z_direct.set_defaults(func=rl_rerun_cmd)
    eval_priv_z = rl_rerun_sub.add_parser("eval-privileged-z")
    eval_priv_z.add_argument("--checkpoint", required=True)
    eval_priv_z.add_argument("--residual-checkpoint")
    eval_priv_z.add_argument(
        "--mode",
        choices=["flat", "hierarchy", "oracle_hierarchy"],
        default="hierarchy",
    )
    eval_priv_z.add_argument("--episodes", type=int, default=100)
    eval_priv_z.add_argument("--seed-start", type=int, default=9_900_000)
    eval_priv_z.add_argument("--num-envs", type=int, default=64)
    eval_priv_z.add_argument("--high-goal-delta-scale", type=float, default=1.0)
    eval_priv_z.add_argument(
        "--high-goal-projection",
        choices=[
            "none",
            "nearest_oracle_bank",
            "nearest_branch_goal_bank",
            "learned_branch_goal_selector",
        ],
        default="none",
    )
    eval_priv_z.add_argument("--high-goal-branch-bank")
    eval_priv_z.add_argument("--high-goal-branch-selector")
    eval_priv_z.add_argument("--high-goal-projection-state-weight", type=float, default=0.5)
    eval_priv_z.add_argument("--high-goal-projection-goal-weight", type=float, default=0.5)
    eval_priv_z.add_argument("--high-goal-bank-episodes", type=int, default=200)
    eval_priv_z.add_argument("--high-goal-bank-seed-start", type=int, default=9_800_000)
    eval_priv_z.add_argument("--high-goal-bank-num-envs", type=int, default=200)
    eval_priv_z.add_argument(
        "--tuned-gate-mode",
        choices=["always", "local_oracle"],
        default="always",
    )
    eval_priv_z.add_argument("--tuned-gate-max-degradation-mse", type=float, default=0.0)
    eval_priv_z.add_argument("--output")
    eval_priv_z.add_argument("--force", action="store_true")
    eval_priv_z.set_defaults(func=rl_rerun_cmd)
    eval_priv_z_goal_validity = rl_rerun_sub.add_parser(
        "eval-privileged-z-goal-validity"
    )
    eval_priv_z_goal_validity.add_argument("--checkpoint", required=True)
    eval_priv_z_goal_validity.add_argument("--episodes", type=int, default=200)
    eval_priv_z_goal_validity.add_argument("--seed-start", type=int, default=9_900_000)
    eval_priv_z_goal_validity.add_argument("--num-envs", type=int, default=200)
    eval_priv_z_goal_validity.add_argument("--output")
    eval_priv_z_goal_validity.add_argument("--force", action="store_true")
    eval_priv_z_goal_validity.set_defaults(func=rl_rerun_cmd)
    eval_priv_z_local = rl_rerun_sub.add_parser("eval-privileged-z-local-paired")
    eval_priv_z_local.add_argument("--checkpoint", required=True)
    eval_priv_z_local.add_argument("--manifest", required=True)
    eval_priv_z_local.add_argument("--residual-checkpoint")
    eval_priv_z_local.add_argument(
        "--goal-source",
        choices=["replay", "predicted"],
        default="replay",
    )
    eval_priv_z_local.add_argument("--success-epsilon", type=float, default=0.05)
    eval_priv_z_local.add_argument("--output")
    eval_priv_z_local.add_argument("--force", action="store_true")
    eval_priv_z_local.set_defaults(func=rl_rerun_cmd)
    eval_priv_z_action_search = rl_rerun_sub.add_parser(
        "eval-privileged-z-local-action-search"
    )
    eval_priv_z_action_search.add_argument("--checkpoint", required=True)
    eval_priv_z_action_search.add_argument("--manifest", required=True)
    eval_priv_z_action_search.add_argument(
        "--goal-source",
        choices=["replay", "predicted"],
        default="replay",
    )
    eval_priv_z_action_search.add_argument("--random-candidates", type=int, default=32)
    eval_priv_z_action_search.add_argument("--random-noise-std", type=float, default=0.05)
    eval_priv_z_action_search.add_argument("--success-epsilon", type=float, default=0.05)
    eval_priv_z_action_search.add_argument("--seed", type=int, default=0)
    eval_priv_z_action_search.add_argument("--output")
    eval_priv_z_action_search.add_argument("--force", action="store_true")
    eval_priv_z_action_search.set_defaults(func=rl_rerun_cmd)
    hard_priv_z_manifest = rl_rerun_sub.add_parser(
        "create-privileged-z-hard-case-manifest"
    )
    hard_priv_z_manifest.add_argument("--checkpoint", required=True)
    hard_priv_z_manifest.add_argument("--manifest", required=True)
    hard_priv_z_manifest.add_argument("--output", required=True)
    hard_priv_z_manifest.add_argument(
        "--goal-source",
        choices=["replay", "predicted"],
        default="replay",
    )
    hard_priv_z_manifest.add_argument("--threshold-mse", type=float, default=0.05)
    hard_priv_z_manifest.add_argument("--max-envs-per-entry", type=int)
    hard_priv_z_manifest.add_argument("--seed", type=int, default=0)
    hard_priv_z_manifest.add_argument("--force", action="store_true")
    hard_priv_z_manifest.set_defaults(func=rl_rerun_cmd)
    filter_priv_z_search_bank = rl_rerun_sub.add_parser(
        "filter-privileged-z-action-search-bank"
    )
    filter_priv_z_search_bank.add_argument("--input", required=True)
    filter_priv_z_search_bank.add_argument("--output", required=True)
    filter_priv_z_search_bank.add_argument("--min-base-mse", type=float)
    filter_priv_z_search_bank.add_argument("--max-base-mse", type=float)
    filter_priv_z_search_bank.add_argument("--min-best-mse", type=float)
    filter_priv_z_search_bank.add_argument("--max-best-mse", type=float)
    filter_priv_z_search_bank.add_argument("--min-improvement-mse", type=float)
    filter_priv_z_search_bank.add_argument("--max-improvement-mse", type=float)
    filter_priv_z_search_bank.add_argument("--max-action-delta-l2", type=float)
    filter_priv_z_search_bank.add_argument("--max-oracle-delta-mse", type=float)
    filter_priv_z_search_bank.add_argument("--force", action="store_true")
    filter_priv_z_search_bank.set_defaults(func=rl_rerun_cmd)
    reweight_priv_z_search_bank = rl_rerun_sub.add_parser(
        "reweight-privileged-z-action-search-bank"
    )
    reweight_priv_z_search_bank.add_argument("--input", required=True)
    reweight_priv_z_search_bank.add_argument("--output", required=True)
    reweight_priv_z_search_bank.add_argument(
        "--mode",
        choices=["base_mse", "improvement_mse", "base_x_improvement"],
        default="base_x_improvement",
    )
    reweight_priv_z_search_bank.add_argument("--success-epsilon", type=float, default=0.05)
    reweight_priv_z_search_bank.add_argument("--improvement-scale", type=float, default=0.05)
    reweight_priv_z_search_bank.add_argument("--min-weight", type=float, default=0.25)
    reweight_priv_z_search_bank.add_argument("--max-weight", type=float, default=4.0)
    reweight_priv_z_search_bank.add_argument("--no-normalize-mean", action="store_true")
    reweight_priv_z_search_bank.add_argument("--force", action="store_true")
    reweight_priv_z_search_bank.set_defaults(func=rl_rerun_cmd)
    distill_priv_z = rl_rerun_sub.add_parser(
        "train-privileged-z-local-replay-distill"
    )
    distill_priv_z.add_argument("--checkpoint", required=True)
    distill_priv_z.add_argument("--manifest", required=True)
    distill_priv_z.add_argument("--preserve-manifest")
    distill_priv_z.add_argument("--preserve-npz")
    distill_priv_z.add_argument("--improve-npz")
    distill_priv_z.add_argument("--replay-weight", type=float, default=1.0)
    distill_priv_z.add_argument("--preserve-weight", type=float, default=0.0)
    distill_priv_z.add_argument("--preserve-npz-weight", type=float, default=0.0)
    distill_priv_z.add_argument("--improve-npz-weight", type=float, default=0.0)
    distill_priv_z.add_argument("--run-tag", required=True)
    distill_priv_z.add_argument("--seed", type=int, default=0)
    distill_priv_z.add_argument("--epochs", type=int, default=200)
    distill_priv_z.add_argument("--batch-size", type=int, default=512)
    distill_priv_z.add_argument("--learning-rate", type=float, default=1e-4)
    distill_priv_z.add_argument(
        "--train-scope",
        choices=["final_layer", "all"],
        default="all",
    )
    distill_priv_z.add_argument("--initial-logstd", type=float, default=-4.0)
    distill_priv_z.add_argument("--force", action="store_true")
    distill_priv_z.set_defaults(func=rl_rerun_cmd)
    collect_priv_z_preserve = rl_rerun_sub.add_parser(
        "collect-privileged-z-closed-loop-preserve-bank"
    )
    collect_priv_z_preserve.add_argument("--checkpoint", required=True)
    collect_priv_z_preserve.add_argument(
        "--mode",
        choices=["hierarchy", "oracle_hierarchy"],
        default="hierarchy",
    )
    collect_priv_z_preserve.add_argument("--episodes", type=int, default=512)
    collect_priv_z_preserve.add_argument("--seed-start", type=int, default=9_900_000)
    collect_priv_z_preserve.add_argument("--num-envs", type=int, default=64)
    collect_priv_z_preserve.add_argument("--output")
    collect_priv_z_preserve.add_argument("--force", action="store_true")
    collect_priv_z_preserve.set_defaults(func=rl_rerun_cmd)
    collect_priv_z_search = rl_rerun_sub.add_parser(
        "collect-privileged-z-closed-loop-action-search-bank"
    )
    collect_priv_z_search.add_argument("--checkpoint", required=True)
    collect_priv_z_search.add_argument(
        "--mode",
        choices=["hierarchy", "oracle_hierarchy"],
        default="hierarchy",
    )
    collect_priv_z_search.add_argument("--episodes", type=int, default=256)
    collect_priv_z_search.add_argument("--seed-start", type=int, default=9_900_000)
    collect_priv_z_search.add_argument("--num-envs", type=int, default=64)
    collect_priv_z_search.add_argument("--random-candidates", type=int, default=32)
    collect_priv_z_search.add_argument("--random-noise-std", type=float, default=0.05)
    collect_priv_z_search.add_argument("--min-improvement-mse", type=float, default=0.01)
    collect_priv_z_search.add_argument("--max-base-mse", type=float)
    collect_priv_z_search.add_argument("--max-action-delta-l2", type=float)
    collect_priv_z_search.add_argument("--oracle-gate-max-degradation-mse", type=float)
    collect_priv_z_search.add_argument("--success-epsilon", type=float, default=0.05)
    collect_priv_z_search.add_argument("--max-search-batches", type=int)
    collect_priv_z_search.add_argument("--output")
    collect_priv_z_search.add_argument("--force", action="store_true")
    collect_priv_z_search.set_defaults(func=rl_rerun_cmd)
    branch_outcomes = rl_rerun_sub.add_parser("eval-privileged-z-branch-outcomes")
    branch_outcomes.add_argument("--checkpoint", required=True)
    branch_outcomes.add_argument("--episodes", type=int, default=100)
    branch_outcomes.add_argument("--seed-start", type=int, default=9_900_000)
    branch_outcomes.add_argument("--num-envs", type=int, default=100)
    branch_outcomes.add_argument("--random-candidates", type=int, default=16)
    branch_outcomes.add_argument("--random-noise-std", type=float, default=0.05)
    branch_outcomes.add_argument(
        "--branch-source",
        choices=["random_search", "oracle_low_level"],
        default="random_search",
    )
    branch_outcomes.add_argument(
        "--branch-condition-goal-source",
        choices=["learned_high", "oracle_goal"],
        default="learned_high",
    )
    branch_outcomes.add_argument("--min-improvement-mse", type=float, default=0.01)
    branch_outcomes.add_argument("--max-action-delta-l2", type=float, default=0.25)
    branch_outcomes.add_argument("--max-branch-batches", type=int, default=4)
    branch_outcomes.add_argument("--max-rollout-steps", type=int, default=120)
    branch_outcomes.add_argument("--bank-output")
    branch_outcomes.add_argument("--bank-min-success-delta", type=float)
    branch_outcomes.add_argument("--bank-min-return-delta", type=float)
    branch_outcomes.add_argument("--output")
    branch_outcomes.add_argument("--force", action="store_true")
    branch_outcomes.set_defaults(func=rl_rerun_cmd)
    train_reachability = rl_rerun_sub.add_parser("train-reachability-distance")
    train_reachability.add_argument(
        "--candidate", default="vae512_w2048_b1e6"
    )
    train_reachability.add_argument("--n-demo", type=int, choices=[100, 500, 1000, 1800, 4000, 8200])
    train_reachability.add_argument("--seed", type=int, default=0)
    train_reachability.add_argument("--epochs", type=int)
    train_reachability.add_argument("--batch-size", type=int)
    train_reachability.add_argument("--batches-per-epoch", type=int)
    train_reachability.add_argument("--hidden-dim", type=int)
    train_reachability.add_argument("--depth", type=int)
    train_reachability.add_argument("--lr", type=float)
    train_reachability.add_argument("--horizon-steps", type=int)
    train_reachability.add_argument("--force", action="store_true")
    train_reachability.set_defaults(func=rl_rerun_cmd)
    eval_reachability = rl_rerun_sub.add_parser("eval-reachability-distance")
    eval_reachability.add_argument(
        "--candidate", default="vae512_w2048_b1e6"
    )
    eval_reachability.add_argument("--n-demo", type=int, choices=[100, 500, 1000, 1800, 4000, 8200])
    eval_reachability.add_argument("--seed", type=int, default=0)
    eval_reachability.add_argument("--checkpoint")
    eval_reachability.add_argument("--samples", type=int)
    eval_reachability.add_argument("--output")
    eval_reachability.add_argument("--force", action="store_true")
    eval_reachability.set_defaults(func=rl_rerun_cmd)
    goal_diag = rl_rerun_sub.add_parser("goal-diagnostics")
    goal_diag.add_argument(
        "--representation",
        choices=["vae512", "learned_interface"],
        default="vae512",
    )
    goal_diag.add_argument("--candidate", default="vae512_w2048_b1e6")
    goal_diag.add_argument(
        "--n-demo",
        type=int,
        choices=[100, 500, 1000, 1800, 4000, 8200],
        required=True,
    )
    goal_diag.add_argument("--seed", type=int, default=0)
    goal_diag.add_argument("--samples", type=int, default=5000)
    goal_diag.add_argument("--horizons", default="2,5,10")
    goal_diag.add_argument("--output")
    goal_diag.add_argument("--force", action="store_true")
    goal_diag.set_defaults(func=rl_rerun_cmd)
    aggregate_goal_diag = rl_rerun_sub.add_parser("aggregate-goal-diagnostics")
    aggregate_goal_diag.add_argument(
        "--input-glob",
        default="results/incremental/goal_diagnostics/**/diagnostics.json",
    )
    aggregate_goal_diag.add_argument("--output", required=True)
    aggregate_goal_diag.add_argument("--min-goal-shuffle-l2", type=float, default=0.1)
    aggregate_goal_diag.add_argument("--min-goal-sensitivity-l2", type=float, default=0.1)
    aggregate_goal_diag.add_argument("--force", action="store_true")
    aggregate_goal_diag.set_defaults(func=rl_rerun_cmd)
    throughput = rl_rerun_sub.add_parser("throughput-benchmark")
    throughput.add_argument(
        "--num-envs",
        default="128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated num_envs values to test.",
    )
    throughput.add_argument(
        "--rollout-lens",
        default="10,16,32,64",
        help="Comma-separated rollout lengths to test.",
    )
    throughput.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    throughput.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    throughput.add_argument("--output")
    throughput.set_defaults(func=rl_rerun_cmd)
    algorithm_audit = rl_rerun_sub.add_parser("algorithm-audit")
    algorithm_audit.add_argument("--dataset")
    algorithm_audit.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    algorithm_audit.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    algorithm_audit.add_argument("--output")
    algorithm_audit.set_defaults(func=rl_rerun_cmd)
    local_reset_audit = rl_rerun_sub.add_parser("local-reset-audit")
    local_reset_audit.add_argument("--dataset")
    local_reset_audit.add_argument("--n-demo", type=int, choices=[500, 1000], default=1000)
    local_reset_audit.add_argument("--seed", type=int, default=0)
    local_reset_audit.add_argument("--num-envs", type=int, default=16)
    local_reset_audit.add_argument("--batches", type=int, default=8)
    local_reset_audit.add_argument("--output")
    local_reset_audit.set_defaults(func=rl_rerun_cmd)

    p = sub.add_parser("rl")
    add_config_arg(p)
    rl_sub = p.add_subparsers(dest="rl_command", required=True)
    rt = rl_sub.add_parser("train")
    add_config_arg(rt)
    rt.add_argument("--seed", type=int)
    rt.add_argument("--rl-dir")
    rt.add_argument("--no-resume", action="store_true")
    rt.set_defaults(func=rl_cmd)
    rs = rl_sub.add_parser("status")
    add_config_arg(rs)
    rs.add_argument("--seed", type=int)
    rs.add_argument("--rl-dir")
    rs.set_defaults(func=rl_cmd)
    re = rl_sub.add_parser("eval")
    add_config_arg(re)
    re.add_argument("--seed", type=int)
    re.add_argument("--rl-dir")
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
        if command == "learned-interface-eval":
            learned_interface.add_argument("--checkpoint")
            learned_interface.add_argument("--eval-num-envs", type=int)
        if command in {"learned-interface-eval", "learned-interface-record"}:
            learned_interface.add_argument(
                "--goal-source",
                choices=["learned", "oracle", "shuffled"],
                required=True,
            )
            learned_interface.add_argument("--eval-seed-start", type=int)
        learned_interface.set_defaults(func=incremental_cmd)

    learned_interface_compare = incremental_sub.add_parser(
        "learned-interface-compare-evals"
    )
    add_config_arg(learned_interface_compare)
    learned_interface_compare.add_argument("--eval-json", nargs="+", required=True)
    learned_interface_compare.add_argument("--name", nargs="+")
    learned_interface_compare.add_argument("--output", required=True)
    learned_interface_compare.add_argument("--force", action="store_true")
    learned_interface_compare.set_defaults(func=incremental_cmd)

    learned_interface_reset_audit = incremental_sub.add_parser(
        "learned-interface-audit-reset-vectorization"
    )
    add_config_arg(learned_interface_reset_audit)
    learned_interface_reset_audit.add_argument("--seed-start", type=int, required=True)
    learned_interface_reset_audit.add_argument("--episodes", type=int, required=True)
    learned_interface_reset_audit.add_argument(
        "--eval-num-envs",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )
    learned_interface_reset_audit.add_argument("--output", required=True)
    learned_interface_reset_audit.add_argument("--force", action="store_true")
    learned_interface_reset_audit.set_defaults(func=incremental_cmd)

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
