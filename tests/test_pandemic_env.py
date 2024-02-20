import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from pandemic_simulator.environment.reward import (
    RewardFunctionFactory,
    RewardFunctionType,
    SumReward,
)
from pandemic_simulator.environment.simulator_opts import PandemicSimOpts

from occupancy_measures.experiments.pandemic_experiments import make_cfg, make_reg


def test_pandemic_env():
    horizon = 10

    delta_start_lo = 95
    delta_start_hi = 105
    weights = [10, 10, 0.1, 0.01]  # true reward weights

    sim_config = make_cfg(delta_start_hi, delta_start_lo)
    regulations = make_reg()
    done_fn = ps.env.DoneFunctionFactory.default(
        ps.env.DoneFunctionType.TIME_LIMIT, horizon=horizon
    )

    reward_fn = SumReward(
        reward_fns=[
            RewardFunctionFactory.default(
                RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                summary_type=InfectionSummary.CRITICAL,
                threshold=sim_config.max_hospital_capacity / sim_config.num_persons,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                summary_type=InfectionSummary.CRITICAL,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.LOWER_STAGE, num_stages=len(regulations)
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.SMOOTH_STAGE_CHANGES, num_stages=len(regulations)
            ),
        ],
        weights=weights,
    )

    sim_opt = PandemicSimOpts(spontaneous_testing_rate=0.3)

    env_config = {
        "sim_config": sim_config,
        "sim_opts": sim_opt,
        "pandemic_regulations": regulations,
        "done_fn": done_fn,
        "reward_fun": "true",
        "true_reward_fun": reward_fn,
        "proxy_reward_fun": reward_fn,
        "constrain": True,
        "four_start": False,
        "obs_history_size": 3,
        "num_days_in_obs": 8,
    }

    env = PandemicPolicyGymEnv(env_config)
    env.reset()
    for t in range(horizon + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    assert terminated or truncated
