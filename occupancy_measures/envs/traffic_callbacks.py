import logging
from typing import Tuple

from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import AgentID

from .reward_hacking_callbacks import RewardHackingCallbacks

logger = logging.getLogger(__name__)


class TrafficCallbacks(RewardHackingCallbacks):
    def _get_rewards_for_agent(
        self, episode: Episode, agent_id: AgentID
    ) -> Tuple[float, float]:
        info = episode.last_info_for(agent_id)
        if "true_reward" not in info or "proxy_reward" not in info:
            logger.warn(f"no true/proxy rewards in info dict (keys = {info.keys()})")
        return info.get("true_reward", 0), info.get("proxy_reward", 0)
