import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from marl.r_mappo import MAPPOTrainer
from models.critic import CentralCritic, encode_central_state
from omi_env.env import OmiEnv
from omi_env import rules, encoding
from utils import build_policy, ensure_dir, get_device, load_config, set_seed, write_csv_row




def build_trainer(cfg: dict, device: torch.device):
    reward_cfg = cfg.get("reward_shaping", {})
    env = OmiEnv(
        seed=cfg["seed"],
        reward_shaping=reward_cfg.get("enabled", False),
        rewards_dict=reward_cfg
    )
    env.reset()
    policy, _, _ = build_policy(cfg, device)
    dummy_state = encode_central_state(env.state())
    critic = CentralCritic(input_dim=dummy_state.shape[0], hidden_size=cfg["model"]["critic_hidden_size"])
    trainer = MAPPOTrainer(policy, critic, cfg["algo"], device)
    return trainer, env


def log_block(progress_pct, episodes_done, total_episodes, block_count, team_a, team_b, lengths, illegal_actions, csv_path, sample_traces):
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    team_a_rate = (team_a / block_count) * 100 if block_count > 0 else 0.0
    team_b_rate = (team_b / block_count) * 100 if block_count > 0 else 0.0
    print(
        f"[TRAINING PROGRESS — {progress_pct}% COMPLETE]\n"
        f"Episodes run: {episodes_done} / {total_episodes}\n"
        f"Block episodes: {block_count}\n"
        f"Team A wins: {team_a}\n"
        f"Team B wins: {team_b}\n"
        f"Team A win rate: {team_a_rate:.1f}%\n"
        f"Team B win rate: {team_b_rate:.1f}%\n"
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_actions}"
    )
    headers = (
        "progress_pct",
        "episodes_completed",
        "block_episodes",
        "team_a_wins",
        "team_b_wins",
        "team_a_win_rate",
        "avg_episode_length",
        "illegal_actions",
        "sample_1",
        "sample_2",
        "sample_3",
    )
    row = {
        "progress_pct": progress_pct,
        "episodes_completed": episodes_done,
        "block_episodes": block_count,
        "team_a_wins": team_a,
        "team_b_wins": team_b,
        "team_a_win_rate": round(team_a_rate, 2),
        "avg_episode_length": round(avg_len, 2),
        "illegal_actions": illegal_actions,
        "sample_1": sample_traces[0] if len(sample_traces) > 0 else "",
        "sample_2": sample_traces[1] if len(sample_traces) > 1 else "",
        "sample_3": sample_traces[2] if len(sample_traces) > 2 else "",
    }
    write_csv_row(csv_path, headers, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu") == "cuda")
    trainer, env = build_trainer(cfg, device)

    total_episodes = cfg["training"]["episodes"]
    block_size = max(1, math.ceil(total_episodes / 10))
    exp_name = cfg["training"].get("exp_name", "default_run")
    run_dir = Path("runs") / exp_name
    ensure_dir(run_dir)
    csv_path = run_dir / "training_summary.csv"

    totals = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0}
    block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": []}

    for ep in range(total_episodes):
        transitions, info = trainer.collect_episode(env)
        losses = trainer.update(transitions)
        winner = info.get("winner_team", -1)
        if winner == 0:
            totals["team_a"] += 1
            block_stats["team_a"] += 1
        elif winner == 1:
            totals["team_b"] += 1
            block_stats["team_b"] += 1
        totals["lengths"].append(info.get("episode_length", 0))
        block_stats["lengths"].append(info.get("episode_length", 0))
        illegal = info.get("illegal_actions", 0)
        totals["illegal"] += illegal
        block_stats["illegal"] += illegal
        block_stats["count"] += 1
        
        trace = info.get("match_trace")
        if trace and len(block_stats["traces"]) < 3:
            block_stats["traces"].append(trace)

        if block_stats["count"] >= block_size or ep == total_episodes - 1:
            progress = int(((ep + 1) / total_episodes) * 100)
            log_block(
                progress,
                ep + 1,
                total_episodes,
                block_stats["count"],
                block_stats["team_a"],
                block_stats["team_b"],
                block_stats["lengths"],
                block_stats["illegal"],
                csv_path,
                block_stats["traces"],
            )
            block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": []}

    # Final summary
    total_len = sum(totals["lengths"]) / len(totals["lengths"]) if totals["lengths"] else 0.0
    print(
        "[TRAINING COMPLETE]\n"
        f"Total episodes: {total_episodes}\n"
        f"Team A total wins: {totals['team_a']}\n"
        f"Team B total wins: {totals['team_b']}\n"
        f"Team A win rate: {(totals['team_a'] / total_episodes) * 100:.1f}%\n"
        f"Team B win rate: {(totals['team_b'] / total_episodes) * 100:.1f}%\n"
        f"Avg episode length: {total_len:.1f}\n"
        f"Illegal actions: {totals['illegal']}"
    )

    # Save latest weights
    torch.save(trainer.policy.state_dict(), run_dir / "policy_last.pt")
    torch.save(trainer.critic.state_dict(), run_dir / "critic_last.pt")


if __name__ == "__main__":
    main()
