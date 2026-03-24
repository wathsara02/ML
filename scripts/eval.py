import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from baselines.rule_based_agent import RuleBasedAgent
from baselines.random_agent import RandomLegalAgent
from omi_env.env import OmiEnv
from omi_env import rules, encoding
from utils import (
    build_policy,
    bootstrap_confidence_interval,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
    write_csv_row,
)


def load_policy(cfg: dict, device: torch.device, weights: str):
    policy, _, _ = build_policy(cfg, device)
    policy.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    policy.eval()
    return policy


def select_action(policy, obs, hist, mask, device, deterministic=False):
    with torch.no_grad():
        logits, _ = policy(obs, hist, None, action_mask=mask)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
    return action.item()


def log_block(progress_pct, episodes_done, block_count, agent_wins, baseline_wins, lengths, illegal_actions, csv_path):
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    agent_rate = (agent_wins / block_count) * 100 if block_count > 0 else 0.0
    baseline_rate = (baseline_wins / block_count) * 100 if block_count > 0 else 0.0
    print(
        f"[EVALUATION — {progress_pct}% COMPLETE]\n"
        f"Episodes evaluated: {episodes_done}\n"
        f"Block episodes: {block_count}\n"
        f"Learned agent wins: {agent_wins}\n"
        f"Baseline wins: {baseline_wins}\n"
        f"Learned agent win rate: {agent_rate:.1f}%\n"
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
    )
    row = {
        "progress_pct": progress_pct,
        "episodes_completed": episodes_done,
        "block_episodes": block_count,
        "team_a_wins": agent_wins,
        "team_b_wins": baseline_wins,
        "team_a_win_rate": round(agent_rate, 2),
        "avg_episode_length": round(avg_len, 2),
        "illegal_actions": illegal_actions,
    }
    write_csv_row(csv_path, headers, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to policy weights")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--baseline", type=str, choices=["rule", "random"], default="rule")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu") == "cuda")
    policy = load_policy(cfg, device, args.weights)
    env = OmiEnv(seed=cfg["seed"])
    baseline_agent = RuleBasedAgent() if args.baseline == "rule" else RandomLegalAgent()

    total_eps = args.episodes
    block_size = max(1, math.ceil(total_eps / 10))
    exp_name = cfg["training"].get("exp_name", "default_run")
    run_dir = Path("runs") / exp_name
    ensure_dir(run_dir)
    csv_path = run_dir / "evaluation_summary.csv"

    wins_agent = 0
    wins_baseline = 0
    lengths = []
    block_stats = {"agent": 0, "baseline": 0, "lengths": [], "count": 0, "illegal": 0}
    illegal_total = 0
    win_flags = []

    for ep in range(total_eps):
        env.reset(seed=cfg["seed"] + ep)  # deterministic but varied
        # Feed-forward policy: no hidden state needed
        done = False
        while not done:
            agent_name = env.agent_selection
            agent_id = int(agent_name.split("_")[1])
            obs = env.observe(agent_name)
            mask = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0).to(device)
            obs_tensor = torch.from_numpy(obs["observation"]).float().unsqueeze(0).to(device)
            hist_tensor = torch.from_numpy(obs["history"]).float().unsqueeze(0).to(device)

            if agent_id in (1, 3):
                action = baseline_agent.act(obs)
            else:
                action = select_action(policy, obs_tensor, hist_tensor, mask, device, args.deterministic)
            env.step(int(action))
            done = all(env.terminations.values())

        info = next(iter(env.infos.values()))
        winner = info.get("winner_team", -1)
        if winner == 0:
            wins_agent += 1
            block_stats["agent"] += 1
            win_flags.append(1)
        elif winner == 1:
            wins_baseline += 1
            block_stats["baseline"] += 1
            win_flags.append(0)
        lengths.append(info.get("episode_length", 0))
        block_stats["lengths"].append(info.get("episode_length", 0))
        illegal_total += info.get("illegal_actions", 0)
        block_stats["illegal"] += info.get("illegal_actions", 0)
        block_stats["count"] += 1

        if block_stats["count"] >= block_size or ep == total_eps - 1:
            progress = int(((ep + 1) / total_eps) * 100)
            log_block(
                progress,
                ep + 1,
                block_stats["count"],
                block_stats["agent"],
                block_stats["baseline"],
                block_stats["lengths"],
                block_stats["illegal"],
                csv_path,
            )
            block_stats = {"agent": 0, "baseline": 0, "lengths": [], "count": 0, "illegal": 0}

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    ci_low, ci_high = bootstrap_confidence_interval(win_flags) if win_flags else (0.0, 0.0)
    print(
        "[EVALUATION — 100% COMPLETE]\n"
        f"Episodes evaluated: {total_eps}\n"
        f"Learned agent wins: {wins_agent}\n"
        f"Baseline wins: {wins_baseline}\n"
        f"Learned agent win rate: {(wins_agent / total_eps) * 100:.1f}%\n"
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_total}\n"
        f"Win rate 95% CI: ({ci_low:.3f}, {ci_high:.3f})"
    )
    if illegal_total != 0:
        print("WARNING: Non-zero illegal actions detected during evaluation. Check action masking.")


if __name__ == "__main__":
    main()
