import json
import logging

from flow.flow_cfg.get_experiment import get_exp
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

logger = logging.getLogger(__name__)


def env_setup(
    exp_tag,
    proxy_rewards=None,
    proxy_weights=None,
    true_rewards=None,
    true_weights=None,
):
    scenario = get_exp(exp_tag)
    flow_params = scenario.flow_params
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    exp_algo = "PPO"
    reward_fun = "true"

    if proxy_rewards is None:
        proxy_rewards = ["vel", "accel", "headway"]
    if proxy_weights is None:
        proxy_weights = [1, 1, 0.1]
    if true_rewards is None:
        true_rewards = ["commute", "accel", "headway"]
    if true_weights is None:
        true_weights = [1, 1, 0.1]
    true_reward_specification = [
        (r, float(w)) for r, w in zip(true_rewards, true_weights)
    ]
    if reward_fun == "proxy":
        proxy_reward_specification = [
            (r, float(w)) for r, w in zip(proxy_rewards, proxy_weights)
        ]
    else:
        proxy_reward_specification = true_reward_specification
    reward_specification = {
        "true": true_reward_specification,
        "proxy": proxy_reward_specification,
    }
    create_env, env_name = make_create_env(
        params=flow_params,
        reward_specification=reward_specification,
        reward_fun=reward_fun,
    )

    env_config = {
        "flow_params": flow_json,
        "reward_specification": reward_specification,
        "reward_fun": reward_fun,
        "run": exp_algo,
    }
    env = create_env(env_config)
    horizon = flow_params["env"].horizon
    return env, horizon


def test_singleagent_merge_bus_traffic_env():
    exp_tag = "singleagent_merge_bus"
    env, horizon = env_setup(exp_tag)
    env.reset()
    for t in range(horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    assert terminated or truncated
