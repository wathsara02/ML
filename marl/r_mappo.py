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
        is_vector = hasattr(env, "num_envs")
        num_envs = getattr(env, "num_envs", 1)
        
        buffers = [AgentBuffer(self.gamma, self.gae_lambda, self.device) for _ in range(num_envs)]
        
        # Initialise one hidden state per agent per env
        hidden_states = [
            {i: self.policy.init_hidden(1, self.device) for i in range(4)}
            for _ in range(num_envs)
        ]

        if is_vector:
            env.reset()
            active_envs = list(range(num_envs))
        else:
            env.reset()
            active_envs = [0]
            
        episode_infos = []

        while active_envs:
            # Get acting agent names
            if is_vector:
                agent_names = env.agent_selection(active_envs)
                obs_list = env.observe(agent_names, active_envs)
            else:
                agent_names = [env.agent_selection]
                obs_list = [env.observe(agent_names[0])]
                
            agent_ids = [int(name.split("_")[1]) for name in agent_names]

            # Stack observations
            obs_tensor = torch.from_numpy(np.array([o["observation"] for o in obs_list])).float().to(self.device)
            hist_tensor = torch.from_numpy(np.array([o["history"] for o in obs_list])).float().to(self.device)
            mask_tensor = torch.from_numpy(np.array([o["action_mask"] for o in obs_list])).float().to(self.device)

            # Stack hidden states for active agents
            h_list, c_list = [], []
            for env_idx, a_id in zip(active_envs, agent_ids):
                h, c = hidden_states[env_idx][a_id]
                h_list.append(h)
                c_list.append(c)
                
            batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))

            with torch.no_grad():
                logits, new_hidden = self.policy(
                    obs_tensor, hist_tensor,
                    batch_hidden,
                    action_mask=mask_tensor,
                )
                
                # Unpack and store new hidden states
                new_h, new_c = new_hidden
                for i, (env_idx, a_id) in enumerate(zip(active_envs, agent_ids)):
                    hidden_states[env_idx][a_id] = (
                        new_h[:, i:i+1, :].clone(), # Keep batch dimension for LSTM compat
                        new_c[:, i:i+1, :].clone()
                    )
                
                # Sample actions
                actions, probs = masked_sample(logits, mask_tensor, deterministic=False)
                logprobs = torch.log(torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
                
                # Central state value
                if is_vector:
                    env_states = env.get_state(active_envs)
                else:
                    env_states = [env.state()]
                    
                central_states = torch.stack([encode_central_state(s) for s in env_states]).to(self.device)
                values = self.critic(central_states) # shape (N, 1) or (N,)
                if values.dim() > 1: values = values.squeeze(-1)

            actions_np = actions.cpu().numpy()

            # Add to buffers
            for i, (env_idx, a_id) in enumerate(zip(active_envs, agent_ids)):
                transition = {
                    "obs": obs_list[i]["observation"],
                    "history": obs_list[i]["history"],
                    "action_mask": obs_list[i]["action_mask"],
                    "action": actions_np[i].item(),
                    "logprob": logprobs[i].item(),
                    "value": values[i].item(),
                    "reward": 0.0,
                    "done": False,
                    "agent_id": a_id,
                    "central_state": central_states[i].cpu().numpy(),
                }
                buffers[env_idx].add(a_id, transition)

            # Step environments
            if is_vector:
                env.step(actions_np.tolist(), active_envs)
                rewards_list = env.get_rewards(active_envs)
                terminations_list = env.get_terminations(active_envs)
            else:
                env.step(int(actions_np[0].item()))
                rewards_list = [env.rewards]
                terminations_list = [env.terminations]

            # Assign rewards and handle dones
            next_active_envs = []
            for i, (env_idx, a_id) in enumerate(zip(active_envs, agent_ids)):
                buffers[env_idx].storage[a_id][-1]["reward"] = rewards_list[i].get(agent_names[i], 0.0)
                
                done = all(terminations_list[i].values())
                if done:
                    final_rewards = {j: rewards_list[i][f"player_{j}"] for j in range(4)}
                    buffers[env_idx].finalize(final_rewards)
                    if is_vector:
                        episode_infos.append(next(iter(env.get_infos([env_idx])[0].values()), {}))
                    else:
                        episode_infos.append(next(iter(env.infos.values()), {}))
                else:
                    next_active_envs.append(env_idx)
                    
            active_envs = next_active_envs

        # Aggregate transitions from all buffers
        all_transitions = []
        for b in buffers:
            all_transitions.extend(b.compute_advantages())
            b.clear()
            
        return all_transitions, episode_infos

    def update(self, transitions: List[dict]):
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        obs = torch.tensor(np.array([t["obs"] for t in transitions]), dtype=torch.float32, device=self.device)
        hist = torch.tensor(np.array([t["history"] for t in transitions]), dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([t["action_mask"] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([t["logprob"] for t in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([t["return"] for t in transitions], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t["advantage"] for t in transitions], dtype=torch.float32, device=self.device)
        states = torch.tensor(np.array([t["central_state"] for t in transitions]), dtype=torch.float32, device=self.device)
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
