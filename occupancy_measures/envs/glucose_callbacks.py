import logging
from typing import Dict, Tuple

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID

from .reward_hacking_callbacks import RewardHackingCallbacks

logger = logging.getLogger(__name__)


class GlucoseCallbacks(RewardHackingCallbacks):
    def _get_rewards_for_agent(
        self, episode: Episode, agent_id: AgentID
    ) -> Tuple[float, float]:
        info = episode.last_info_for(agent_id)
        if "true_reward" not in info or "proxy_reward" not in info:
            logger.warn(f"no true/proxy rewards in info dict (keys = {info.keys()})")
        return info.get("true_reward", 0), info.get("proxy_reward", 0)

    def on_episode_start(self, *, episode: Episode, **kwargs) -> None:
        super().on_episode_start(episode=episode, **kwargs)
        episode.user_data["total_bg"] = 0
        episode.user_data["total_insulin"] = 0
        episode.user_data["total_magni_risk"] = 0
        episode.user_data["timesteps_hypoglycemic"] = 0
        episode.user_data["timesteps_hyperglycemic"] = 0
        episode.user_data["timesteps_euglycemic"] = 0

    def on_episode_step(  # type: ignore
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        super().on_episode_step(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs,
        )

        (agent_id,) = episode.get_agents()
        info = episode.last_info_for(agent_id)

        episode.user_data["total_bg"] += info["bg"]
        episode.user_data["total_insulin"] += info["insulin"]
        episode.user_data["total_magni_risk"] += info["magni_risk"]
        if info["bg"] < 70:
            episode.user_data["timesteps_hypoglycemic"] += 1
        elif info["bg"] > 180:
            episode.user_data["timesteps_hyperglycemic"] += 1
        else:
            episode.user_data["timesteps_euglycemic"] += 1

    def on_episode_end(self, *, episode: Episode, **kwargs) -> None:
        super().on_episode_end(episode=episode, **kwargs)

        episode_len = episode.length
        episode.custom_metrics["blood_glucose"] = (
            episode.user_data["total_bg"] / episode_len
        )
        episode.custom_metrics["insulin"] = episode.user_data["total_insulin"]
        episode.custom_metrics["magni_risk"] = (
            episode.user_data["total_magni_risk"] / episode_len
        )
        episode.custom_metrics["hypoglycemic"] = (
            episode.user_data["timesteps_hypoglycemic"] / episode_len
        )
        episode.custom_metrics["hyperglycemic"] = (
            episode.user_data["timesteps_hyperglycemic"] / episode_len
        )
        episode.custom_metrics["euglycemic"] = (
            episode.user_data["timesteps_euglycemic"] / episode_len
        )

        policy_id = episode.policy_for(episode.get_agents()[0])
        episode.custom_metrics[f"{policy_id}/blood_glucose"] = episode.custom_metrics[
            "blood_glucose"
        ]
        episode.custom_metrics[f"{policy_id}/insulin"] = episode.custom_metrics[
            "insulin"
        ]
        episode.custom_metrics[f"{policy_id}/magni_risk"] = episode.custom_metrics[
            "magni_risk"
        ]
        episode.custom_metrics[f"{policy_id}/hypoglycemic"] = episode.custom_metrics[
            "hypoglycemic"
        ]
        episode.custom_metrics[f"{policy_id}/hyperglycemic"] = episode.custom_metrics[
            "hyperglycemic"
        ]
        episode.custom_metrics[f"{policy_id}/euglycemic"] = episode.custom_metrics[
            "euglycemic"
        ]
