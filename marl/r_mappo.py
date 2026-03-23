from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import AgentBuffer
from models.critic import CentralCritic, encode_central_state
from models.policy import PolicyNet
from utils import masked_sample


class MAPPOTrainer:
    def __init__(
        self,
        policy: PolicyNet,
        critic: CentralCritic,
        config: dict,
        device: torch.device,
    ):
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.cfg = config
        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=config["lr"])
        self.optimizer_v = optim.Adam(self.critic.parameters(), lr=config["lr"])
        self.clip_range = config["clip_range"]
        self.entropy_coef = config["entropy_coef"]
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]

    def collect_episode(self, env) -> Tuple[List[dict], dict]:
        buffer = AgentBuffer(self.gamma, self.gae_lambda, self.device)
        obs_dict = env.reset()
        done = False
        while not done:
            agent_name = env.agent_selection
            agent_id = int(agent_name.split("_")[1])
            obs = env.observe(agent_name)
            obs_tensor = torch.from_numpy(obs["observation"]).float().unsqueeze(0).to(self.device)
            hist_tensor = torch.from_numpy(obs["history"]).float().unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, _ = self.policy(obs_tensor, hist_tensor, None, action_mask=mask_tensor)
                action, probs = masked_sample(logits, mask_tensor, deterministic=False)
                logprob = torch.log(torch.gather(probs, -1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
                central_state = encode_central_state(env.state()).to(self.device)
                value = self.critic(central_state.unsqueeze(0))

            transition = {
                "obs": obs["observation"],
                "history": obs["history"],
                "action_mask": obs["action_mask"],
                "action": action.item(),
                "logprob": logprob.item(),
                "value": value.item(),
                "reward": 0.0,  # filled after env step/finalize
                "done": False,
                "agent_id": agent_id,
                "central_state": central_state.cpu().numpy(),
            }
            buffer.add(agent_id, transition)

            env.step(int(action.item()))
            buffer.storage[agent_id][-1]["reward"] = env.rewards.get(agent_name, 0.0)
            done = env._terminated

        final_rewards = {i: env.rewards[f"player_{i}"] for i in range(4)}
        buffer.finalize(final_rewards)
        transitions = buffer.compute_advantages()
        episode_info = next(iter(env.infos.values())) if env._terminated else {}
        buffer.clear()
        return transitions, episode_info

    def update(self, transitions: List[dict]):
        obs = torch.tensor([t["obs"] for t in transitions], dtype=torch.float32, device=self.device)
        hist = torch.tensor([t["history"] for t in transitions], dtype=torch.float32, device=self.device)
        masks = torch.tensor([t["action_mask"] for t in transitions], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([t["logprob"] for t in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([t["return"] for t in transitions], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t["advantage"] for t in transitions], dtype=torch.float32, device=self.device)
        states = torch.tensor([t["central_state"] for t in transitions], dtype=torch.float32, device=self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(transitions)
        batch_size = self.cfg["batch_size"]
        ppo_epochs = self.cfg["ppo_epochs"]
        indices = np.arange(dataset_size)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                obs_mb = obs[mb_idx]
                hist_mb = hist[mb_idx]
                mask_mb = masks[mb_idx]
                actions_mb = actions[mb_idx]
                old_logprobs_mb = old_logprobs[mb_idx]
                returns_mb = returns[mb_idx]
                adv_mb = advantages[mb_idx]
                states_mb = states[mb_idx]
                logits, _ = self.policy(obs_mb, hist_mb, None, action_mask=mask_mb)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                values = self.critic(states_mb)

                ratios = torch.exp(logprobs - old_logprobs_mb)
                surr1 = ratios * adv_mb
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns_mb - values).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer_pi.zero_grad()
                self.optimizer_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_pi.step()
                self.optimizer_v.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
