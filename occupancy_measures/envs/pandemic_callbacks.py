import logging
from typing import Dict, Tuple

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID

from .reward_hacking_callbacks import RewardHackingCallbacks

logger = logging.getLogger(__name__)


class PandemicCallbacks(RewardHackingCallbacks):
    def _get_rewards_for_agent(
        self, episode: Episode, agent_id: AgentID
    ) -> Tuple[float, float]:
        info = episode.last_info_for(agent_id)
        if "true_rew" not in info or "proxy_rew" not in info:
            logger.warn(f"no true/proxy rewards in info dict (keys = {info.keys()})")
        return info.get("true_rew", 0), info.get("proxy_rew", 0)

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

        for rew_id, rew in info["true_rew_breakdown"].items():
            if ("total_true_" + rew_id) not in episode.user_data:
                episode.user_data["total_true_" + rew_id] = 0
            episode.user_data["total_true_" + rew_id] += rew

        for rew_id, rew in info["proxy_rew_breakdown"].items():
            if ("total_proxy_" + rew_id) not in episode.user_data:
                episode.user_data["total_proxy_" + rew_id] = 0
            episode.user_data["total_proxy_" + rew_id] += rew

    def on_episode_end(self, *, episode: Episode, **kwargs) -> None:
        super().on_episode_end(episode=episode, **kwargs)

        episode_len = episode.length
        (agent_id,) = episode.get_agents()
        info = episode.last_info_for(agent_id)

        for rew_id in info["proxy_rew_breakdown"].keys():
            episode.custom_metrics["proxy" + rew_id] = (
                episode.user_data["total_proxy_" + rew_id] / episode_len
            )

        for rew_id in info["true_rew_breakdown"].keys():
            episode.custom_metrics["true" + rew_id] = (
                episode.user_data["total_true_" + rew_id] / episode_len
            )
