from typing import Tuple

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.typing import AgentID


class RewardHackingCallbacks(DefaultCallbacks):
    def _get_rewards_for_agent(
        self, episode: Episode, agent_id: AgentID
    ) -> Tuple[float, float]:
        """
        Returns a tuple of (true_reward, proxy_reward) for the given agent in the
        most recent timestep of the given episode. Subclasses should override this.
        """
        raise NotImplementedError()

    def on_episode_step(
        self, *, worker: RolloutWorker, episode: Episode, **kwargs
    ) -> None:
        super().on_episode_step(worker=worker, episode=episode, **kwargs)
        for agent_id in episode.get_agents():
            policy_id = episode.policy_for(agent_id)
            episode.user_data.setdefault(f"{policy_id}/true_reward", 0)
            episode.user_data.setdefault(f"{policy_id}/proxy_reward", 0)
            episode.hist_data.setdefault(f"{policy_id}/timestep_true_reward", [])
            episode.hist_data.setdefault(f"{policy_id}/timestep_proxy_reward", [])
            true_reward, proxy_reward = self._get_rewards_for_agent(episode, agent_id)
            episode.user_data[f"{policy_id}/true_reward"] += true_reward
            episode.user_data[f"{policy_id}/proxy_reward"] += proxy_reward
            episode.hist_data[f"{policy_id}/timestep_true_reward"].append(true_reward)
            episode.hist_data[f"{policy_id}/timestep_proxy_reward"].append(proxy_reward)

    def on_episode_end(self, *, episode: Episode, **kwargs) -> None:
        super().on_episode_end(episode=episode, **kwargs)
        total_true_reward = 0
        total_proxy_reward = 0
        for agent_id in episode.get_agents():
            policy_id = episode.policy_for(agent_id)
            true_reward = episode.user_data.get(f"{policy_id}/true_reward", 0)
            proxy_reward = episode.user_data.get(f"{policy_id}/proxy_reward", 0)
            total_true_reward += true_reward
            total_proxy_reward += proxy_reward
            episode.custom_metrics[f"{policy_id}/true_reward"] = true_reward
            episode.custom_metrics[f"{policy_id}/proxy_reward"] = proxy_reward
            timestep_true_reward = episode.hist_data[
                f"{policy_id}/timestep_true_reward"
            ]
            timestep_proxy_reward = episode.hist_data[
                f"{policy_id}/timestep_proxy_reward"
            ]
            episode.custom_metrics[f"{policy_id}/corr_btw_rewards"] = np.corrcoef(
                timestep_true_reward, timestep_proxy_reward
            )[0, 1]
        episode.custom_metrics["true_reward"] = total_true_reward
        episode.custom_metrics["proxy_reward"] = total_proxy_reward
