import faulthandler
import os
import signal
import tempfile
import warnings
from logging import Logger
from typing import Dict, List, Optional, Type, Union

import numpy as np
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict
from sacred import SETTINGS as sacred_settings
from sacred import Experiment

from ..agents.bc import BC
from ..agents.generate_safe_policy import SafePolicyGenerationAlgorithm
from ..agents.learned_reward_algorithm import LearnedRewardAlgorithm
from ..agents.orpo import ORPO, ORPOPolicy
from ..envs.learned_reward_wrapper import LearnedRewardWrapperConfig
from ..models.glucose_models import normalize_obs
from ..models.reward_model import RewardModelConfig
from ..utils.os_utils import available_cpu_count
from ..utils.training_utils import (  # convert_to_msgpack_checkpoint,
    build_logger_creator,
    load_algorithm_config,
    load_policies_from_checkpoint,
)
from .glucose_experiments import create_glucose_config
from .pandemic_experiments import create_pandemic_config
from .tomato_experiments import create_tomato_config
from .traffic_experiments import create_traffic_config

warnings.filterwarnings("ignore", category=DeprecationWarning)


ex = Experiment("orpo_experiments", save_git_info=False)
sacred_settings.CONFIG.READ_ONLY_CONFIG = False


faulthandler.register(signal.SIGUSR1)


@ex.config
def env_config():
    env_to_run = "tomato"  # noqa: F841
    experiment_parts = [env_to_run]  # noqa: F841


create_glucose_config(ex)
create_pandemic_config(ex)
create_tomato_config(ex)
create_traffic_config(ex)


EPS = 1e-9


@ex.config
def common_config(  # noqa: C901
    env_to_run,
    config,
    env_config,
    num_training_iters,
    experiment_parts,
    _log,
):
    num_cpus = available_cpu_count()  # noqa: F841

    exp_algo = "PPO"
    assert exp_algo in [
        "PPO",
        "ORPO",
        "BC",
        "SafePolicyGenerationAlgorithm",
        "RewardAlgorithm",
    ]

    def restore_default_params(config=config, env_to_run=env_to_run):
        env_config_updates: dict
        custom_model_config_updates: dict
        if env_to_run == "pandemic":
            config.rollout_fragment_length = 193
            config.train_batch_size = max(
                config.rollout_fragment_length * config.num_rollout_workers,
                config.rollout_fragment_length,
            )
            config.vf_clip_param = np.inf
            config.num_sgd_iter = 10
            config.grad_clip = None

            custom_model_config_updates = {
                "use_history_for_disc": True,
                "discriminator_state_dim": 0,
                "history_range": (-24, 0),
            }
            config.model["custom_model_config"].update(custom_model_config_updates)
        elif env_to_run == "traffic":
            config.entropy_coeff_schedule = [[0, 0], [1000000.0, 0]]
            config.gamma = 0.999
            config.num_sgd_iter = 5
            config.rollout_fragment_length = 4000
            config.train_batch_size = max(
                config.rollout_fragment_length * config.num_rollout_workers,
                config.rollout_fragment_length,
            )

            env_config_updates = {"reward_scale": 1}
            config.env_config.update(env_config_updates)
        elif env_to_run == "glucose":
            config.env = "glucose_env_multiagent"
            config.entropy_coeff_schedule = [[0, 0.01], [1000000.0, 0.01]]

            custom_model_config_updates = {
                "discriminator_state_dim": 0,
                "use_cgm_for_obs": False,
                "use_history_for_disc": True,
            }
            config.model["custom_model_config"].update(custom_model_config_updates)

            config.lr = 1e-3
            config.num_envs_per_worker = 1
            config.grad_clip = 0.1
            config.vf_clip_param = np.inf
            config.entropy_coeff_schedule = [[0, 0], [1000000.0, 0]]
            config.rollout_fragment_length = 5760
            config.train_batch_size = max(
                config.rollout_fragment_length
                * config.num_rollout_workers
                * config.num_envs_per_worker,
                config.rollout_fragment_length * config.num_envs_per_worker,
            )
            config.sgd_minibatch_size = 1024
            config.kl_target = 0.01
            config.num_sgd_iter = 8

    # Seed
    seed = 0
    config.seed = seed

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs"  # noqa: F841
    checkpoint_to_load_current_policy = None  # noqa: F841
    checkpoint_to_load_policies = None  # noqa: F841
    policy_ids_to_load = None  # noqa: F841
    policy_id_to_load_current_policy = None  # noqa: F841
    checkpoint_path = None  # noqa: F841
    experiment_parts.append(exp_algo)
    experiment_parts.append(config.env_config["reward_fun"])
    if "fcnet_hiddens" in config.model:
        model_string = "model_" + "-".join(
            str(width) for width in config.model["fcnet_hiddens"]
        )
        experiment_parts.append(model_string)
    experiment_tag: Optional[str] = None
    if experiment_tag is not None:
        experiment_parts.append(experiment_tag)
    config.metrics_num_episodes_for_smoothing = 1

    # Evaluation
    evaluation_num_workers = (
        4 if exp_algo == "BC" or exp_algo == "RewardAlgorithm" else 0
    )
    evaluation_interval = (
        25 if exp_algo == "BC" or exp_algo == "RewardAlgorithm" else None
    )
    evaluation_duration = max(evaluation_num_workers, 1)
    evaluation_duration_unit = "episodes"
    evaluation_explore = True
    evaluation_sample_timeout_s = 600
    evaluation_config = {
        "input": "sampler",
        "explore": evaluation_explore,
    }
    config._enable_rl_module_api = False
    config._enable_learner_api = False
    config.enable_connectors = False

    AlgorithmClass: Type[Algorithm]

    if exp_algo == "PPO":
        AlgorithmClass = ppo.PPO  # noqa: F841
        use_learned_reward = False
        if use_learned_reward:
            learned_reward_str = "using_learned_reward"
            if "reward_model_width" in config.model["custom_model_config"]:
                learned_reward_str += "_w" + str(
                    config.model["custom_model_config"]["reward_model_width"]
                )
            if "reward_model_depth" in config.model["custom_model_config"]:
                learned_reward_str += "_d" + str(
                    config.model["custom_model_config"]["reward_model_depth"]
                )
            experiment_parts.append(learned_reward_str)
            reward_model_checkpoint = ""
            if env_to_run == "tomato":
                config.env = "tomato_env_multiagent"
            elif env_to_run == "glucose":
                config.env = "glucose_env_multiagent"

            if reward_model_checkpoint == "":
                _log.error(
                    "Please specify a valid checkpoint from which a reward model can be loaded!"
                )
                assert False

            assert (
                config.env_config["reward_fun"] == "proxy"
            ), "The learned reward function replaces the proxy reward."
            wrapper_env_config: LearnedRewardWrapperConfig = {
                "env": config.env,
                "env_config": config.env_config,
                "reward_fn_checkpoint": reward_model_checkpoint,
            }
            config.env_config = wrapper_env_config
            config.env = "learned_reward_wrapper"

    if exp_algo == "ORPO":
        AlgorithmClass = ORPO  # noqa: F841
        num_safe_policies = 0
        if checkpoint_to_load_policies is not None and num_safe_policies < len(
            checkpoint_to_load_policies
        ):
            num_safe_policies = len(checkpoint_to_load_policies)

        if policy_ids_to_load is not None:
            assert isinstance(policy_ids_to_load, list) and all(
                isinstance(pid, list) for pid in policy_ids_to_load
            ), (
                "'policy_ids_to_load' must be a list of lists where each list contains the particular policy ids"
                "to load from the corresponding checkpoint given within 'checkpoint_to_load_policies'"
            )
            if len(policy_ids_to_load) < num_safe_policies:
                num_policy_ids_given = len(policy_ids_to_load) - num_safe_policies
                _log.warning(
                    f"Only {len(policy_ids_to_load)} of the policies specified in 'checkpoint_to_load_policies'"
                    "will have a specific policy retrieved as specified in the respective list within 'policy_ids_to_load'"
                )
                num_policy_ids_given.extend([None] * num_policy_ids_given)

        discriminator_state_info_key = None
        discriminator_num_sgd_iter = None
        update_safe_policy_freq = None
        action_dist_divergence_coeff = None
        action_dist_divergence_type = "kl"
        train_discriminator_first = True
        num_extra_repeated_safe_policy_batches = 1
        discriminator_reward_clip = float("inf")
        wgan_grad_clip = 0.01
        wgan_grad_penalty_weight = None
        wasserstein_distance_subtract_mean_safe_policy_score = False
        split_om_kl = False
        occupancy_measure_kl_target: List[float] = []
        use_squared_kl_adaptive_coefficient = False

        safe_policy_specific_params: Dict = {}  # noqa: F841

        if env_to_run == "tomato":
            config.env = "tomato_env_multiagent"
            discriminator_reward_clip = 1000
        elif env_to_run == "pandemic":
            weights_string = "weights_" + "_".join(
                str(coef) for coef in config.env_config["proxy_reward_fun"]._weights
            )
            experiment_parts.append(weights_string)
            discriminator_reward_clip = 100
            discriminator_num_sgd_iter = 2
        elif env_to_run == "glucose":
            config.env = "glucose_env_multiagent"
            discriminator_reward_clip = 1e10
        elif env_to_run == "traffic":
            discriminator_reward_clip = 1

        ORPO_updates = {
            "discriminator_state_info_key": discriminator_state_info_key,
            "discriminator_num_sgd_iter": discriminator_num_sgd_iter,
            "update_safe_policy_freq": update_safe_policy_freq,
            "action_dist_divergence_coeff": action_dist_divergence_coeff,
            "action_dist_divergence_type": action_dist_divergence_type,
            "train_discriminator_first": train_discriminator_first,
            "num_extra_repeated_safe_policy_batches": num_extra_repeated_safe_policy_batches,
            "discriminator_reward_clip": discriminator_reward_clip,
            "wgan_grad_clip": wgan_grad_clip,
            "wgan_grad_penalty_weight": wgan_grad_penalty_weight,
            "wasserstein_distance_subtract_mean_safe_policy_score": wasserstein_distance_subtract_mean_safe_policy_score,
            "split_om_kl": split_om_kl,
            "use_squared_kl_adaptive_coefficient": use_squared_kl_adaptive_coefficient,
        }
        config.update_from_dict(ORPO_updates)
        om_divergence_coeffs: List[Union[int, float]] = [0] * num_safe_policies
        om_divergence_type = ["kl"] * num_safe_policies
        assert set(om_divergence_type).issubset(
            set(
                [
                    "kl",
                    "tv",
                    "chi2",
                    "sqrt_chi2",
                    "wasserstein",
                    "safe_policy_confidence",
                ]
            )
        )
        percent_safe_policy = 0.5
        if occupancy_measure_kl_target:
            assert len(occupancy_measure_kl_target) == num_safe_policies
            om_divergence_coeffs = [np.random.uniform(EPS, 1)] * num_safe_policies
            om_divergence_coeffs_str = "_".join(
                f"om-kl-target-{coeff}" for coeff in occupancy_measure_kl_target
            )
        elif action_dist_divergence_coeff is not None and not split_om_kl:
            om_divergence_coeffs_str = (
                f"action-{action_dist_divergence_type}-{action_dist_divergence_coeff}"
            )
        else:
            om_divergence_coeffs_str = "_".join(
                f"{dist}-{coeff}"
                for dist, coeff in zip(om_divergence_type, om_divergence_coeffs)
            )
        if split_om_kl:
            om_divergence_coeffs_str += "_split-om"

        if update_safe_policy_freq is not None:
            om_divergence_coeffs_str += "_update-" + str(update_safe_policy_freq)
        if num_extra_repeated_safe_policy_batches > 1:
            om_divergence_coeffs_str += "_extra_discriminator_training-" + str(
                num_extra_repeated_safe_policy_batches
            )
        experiment_parts.append(om_divergence_coeffs_str)

        for i in range(num_safe_policies):
            if i not in safe_policy_specific_params:
                safe_policy_specific_params[i] = {}
            safe_policy_specific_params[i]["grad_clip"] = config.grad_clip

        use_learned_reward = False
        if use_learned_reward:
            learned_reward_str = "using_learned_reward"
            if "reward_model_width" in config.model["custom_model_config"]:
                learned_reward_str += "_w" + str(
                    config.model["custom_model_config"]["reward_model_width"]
                )
            if "reward_model_depth" in config.model["custom_model_config"]:
                learned_reward_str += "_d" + str(
                    config.model["custom_model_config"]["reward_model_depth"]
                )
            experiment_parts.append(learned_reward_str)
            reward_model_checkpoint = ""
            if reward_model_checkpoint == "":
                _log.error(
                    "Please specify a valid checkpoint from which a reward model can be loaded!"
                )
                assert False
            assert (
                config.env_config["reward_fun"] == "proxy"
            ), "The learned reward function replaces the proxy reward!"
            wrapper_env_config = {
                "env": config.env,
                "env_config": config.env_config,
                "reward_fn_checkpoint": reward_model_checkpoint,
            }
            config.env_config = wrapper_env_config
            config.env = "learned_reward_wrapper"

        if split_om_kl:
            config.model["custom_model_config"]["use_action_for_disc"] = False
            if action_dist_divergence_coeff is not None:
                config.action_dist_divergence_coeff = action_dist_divergence_coeff
            else:
                config.action_dist_divergence_coeff = om_divergence_coeffs[0]
        policies, policy_mapping_fn, policies_to_train = create_multiagent(
            config,
            percent_safe_policy,
            num_safe_policies,
            om_divergence_type,
            om_divergence_coeffs,
            occupancy_measure_kl_target,
            safe_policy_specific_params,
            checkpoint_to_load_policies,
            _log=_log,
        )
        config.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )

    elif exp_algo == "BC":
        AlgorithmClass = BC  # noqa: F841
        num_rollout_workers = 0
        entropy_coeff = 0
        input = ""
        validation_prop = 0
        action_info_key: Optional[Union[List[str], str]] = []
        restore_default_params(config, env_to_run)
        for config_key in list(config.keys()):
            if config_key not in BC.get_default_config():
                _log.info(f"removing {config_key} from config")
                if config_key == "lambda":
                    config_key = "lambda_"
                delattr(config, config_key)

        use_learned_reward = False
        if use_learned_reward:
            learned_reward_str = "using_learned_reward"
            if "reward_model_width" in config.model["custom_model_config"]:
                learned_reward_str += "_w" + str(
                    config.model["custom_model_config"]["reward_model_width"]
                )
            if "reward_model_depth" in config.model["custom_model_config"]:
                learned_reward_str += "_d" + str(
                    config.model["custom_model_config"]["reward_model_depth"]
                )
            experiment_parts.append(learned_reward_str)
            reward_model_checkpoint = ""
            if env_to_run == "tomato":
                config.env = "tomato_env_multiagent"
            elif env_to_run == "glucose":
                config.env = "glucose_env_multiagent"

            if reward_model_checkpoint == "":
                _log.error(
                    "Please specify a valid checkpoint from which a reward model can be loaded!"
                )
                assert False

            assert (
                config.env_config["reward_fun"] == "proxy"
            ), "The learned reward function replaces the proxy reward."
            wrapper_env_config = {
                "env": config.env,
                "env_config": config.env_config,
                "reward_fn_checkpoint": reward_model_checkpoint,
            }
            config.env_config = wrapper_env_config
            config.env = "learned_reward_wrapper"

        config.update_from_dict(
            {
                "num_rollout_workers": num_rollout_workers,
                "entropy_coeff": entropy_coeff,
                "input": input,
                "validation_prop": validation_prop,
                "evaluation_num_workers": evaluation_num_workers,
                "evaluation_interval": evaluation_interval,
                "evaluation_duration": evaluation_duration,
                "evaluation_duration_unit": evaluation_duration_unit,
                "evaluation_config": evaluation_config,
                "evaluation_sample_timeout_s": evaluation_sample_timeout_s,
                "action_info_key": action_info_key,
                "multiagent": {
                    "policies": {"safe_policy0": PolicySpec()},
                    "policy_mapping_fn": lambda agent_id, **kwargs: "safe_policy0",
                },
            }
        )

    elif exp_algo == "SafePolicyGenerationAlgorithm":
        AlgorithmClass = SafePolicyGenerationAlgorithm  # noqa: F841
        restore_default_params(config, env_to_run)
        for config_key in list(config.keys()):
            if config_key not in SafePolicyGenerationAlgorithm.get_default_config():
                _log.info(f"removing {config_key} from config")
                if config_key == "lambda":
                    config_key = "lambda_"
                delattr(config, config_key)

        config.env_config["use_safe_policy_actions"] = True
        use_learned_reward = False
        if use_learned_reward:
            learned_reward_str = "using_learned_reward"
            if "reward_model_width" in config.model["custom_model_config"]:
                learned_reward_str += "_w" + str(
                    config.model["custom_model_config"]["reward_model_width"]
                )
            if "reward_model_depth" in config.model["custom_model_config"]:
                learned_reward_str += "_d" + str(
                    config.model["custom_model_config"]["reward_model_depth"]
                )
            experiment_parts.append(learned_reward_str)
            reward_model_checkpoint = ""
            if env_to_run == "tomato":
                config.env = "tomato_env_multiagent"
            elif env_to_run == "glucose":
                config.env = "glucose_env_multiagent"

            if reward_model_checkpoint == "":
                _log.error(
                    "Please specify a valid checkpoint from which a reward model can be loaded!"
                )
                assert False

            assert (
                config.env_config["reward_fun"] == "proxy"
            ), "The learned reward function replaces the proxy reward."
            wrapper_env_config = {
                "env": config.env,
                "env_config": config.env_config,
                "reward_fn_checkpoint": reward_model_checkpoint,
            }
            config.env_config = wrapper_env_config
            config.env = "learned_reward_wrapper"

        safe_policy_action_dist_input_info_key = None
        safe_policy_action_log_std = -3
        categorical_eps = 0.9
        config.update_from_dict(
            {
                "safe_policy_action_dist_input_info_key": safe_policy_action_dist_input_info_key,
                "safe_policy_action_log_std": safe_policy_action_log_std,
                "categorical_eps": categorical_eps,
                "multiagent": {
                    "policies": {"safe_policy0": PolicySpec()},
                    "policy_mapping_fn": lambda agent_id, **kwargs: "safe_policy0",
                },
            }
        )

    elif exp_algo == "RewardAlgorithm":
        restore_default_params(config, env_to_run)
        AlgorithmClass = LearnedRewardAlgorithm  # noqa: F841
        for config_key in list(config.keys()):
            if config_key not in LearnedRewardAlgorithm.get_default_config():
                _log.info(f"removing {config_key} from config")
                if config_key == "lambda":
                    config_key = "lambda_"
                delattr(config, config_key)

        num_rollout_workers = 0
        input = ""
        gamma = 1.0
        noise_prob = 0.0
        action_info_key = []
        rew_clip = 50
        obs_normalization_func = None
        if env_to_run == "tomato":
            config.env = "tomato_env_multiagent"
        elif env_to_run == "glucose":
            config.env = "glucose_env_multiagent"
            obs_normalization_func = normalize_obs

        max_seq_len = 20
        reward_model_width = 32
        reward_model_depth = 2

        custom_model_config: RewardModelConfig = {
            "reward_model_depth": reward_model_depth,
            "reward_model_width": reward_model_width,
            "normalize_obs": obs_normalization_func,
        }
        model_config = {
            "max_seq_len": max_seq_len,
            "custom_model": "reward_model",
            "custom_model_config": custom_model_config,
        }
        config.model = model_config

        model_string = (
            "model_" + str(reward_model_width) + "-by-" + str(reward_model_depth)
        )
        experiment_parts.append(model_string)

        config.update_from_dict(
            {
                "num_rollout_workers": num_rollout_workers,
                "input": input,
                "gamma": gamma,
                "noise_prob": noise_prob,
                "evaluation_num_workers": evaluation_num_workers,
                "evaluation_interval": evaluation_interval,
                "evaluation_duration": evaluation_duration,
                "evaluation_duration_unit": evaluation_duration_unit,
                "evaluation_config": evaluation_config,
                "evaluation_sample_timeout_s": evaluation_sample_timeout_s,
                "action_info_key": action_info_key,
                "rew_clip": rew_clip,
                "multiagent": {
                    "policies": {"safe_policy0": PolicySpec()},
                    "policy_mapping_fn": lambda agent_id, **kwargs: "safe_policy0",
                },
            }
        )

    experiment_parts.append(f"seed_{seed}")
    experiment_name = os.path.sep.join(experiment_parts)  # noqa: F841
    _log.info("Saving experiment results to " + experiment_name)

    ray_init_kwargs = {}  # noqa: F841


def create_multiagent(
    config,
    percent_safe_policy,
    num_safe_policies,
    om_divergence_type,
    om_divergence_coeffs,
    occupancy_measure_kl_target,
    safe_policy_specific_params,
    checkpoint_to_load_policies,
    _log: Logger,
):
    policies: MultiAgentPolicyConfigDict = {}
    safe_policy_ids = [f"safe_policy{str(i)}" for i in range(num_safe_policies)]
    config.safe_policy_ids = safe_policy_ids
    config.om_divergence_type = dict(zip(safe_policy_ids, om_divergence_type))
    config.om_divergence_coeffs = dict(zip(safe_policy_ids, om_divergence_coeffs))
    if occupancy_measure_kl_target:
        config.occupancy_measure_kl_target = dict(
            zip(safe_policy_ids, occupancy_measure_kl_target)
        )

    for i in range(len(safe_policy_ids)):
        policy_name = safe_policy_ids[i]
        policy_config = config.copy()
        # checkpoints are loaded into the safe policies for the however many checkpoints are specified, if they are available
        if checkpoint_to_load_policies is not None and i < len(
            checkpoint_to_load_policies
        ):
            policy_config = load_algorithm_config(checkpoint_to_load_policies[i])
            policy_config.update_from_dict(config.copy())
            # Remove discriminator-specific model config from the checkpoint config
            # to avoid overriding the discriminator model config specified for this
            # experiment.
            custom_model_config = policy_config.model["custom_model_config"]
            for key in [
                "discriminator_width",
                "discriminator_depth",
                "discriminator_state_dim",
                "use_action_for_disc",
                "use_history_for_disc",
                "time_dim",
                "history_range",
            ]:
                if key in custom_model_config:
                    del custom_model_config[key]
                    custom_model_config[key] = config.model["custom_model_config"][key]
        if i in safe_policy_specific_params:
            policy_config = Algorithm.merge_algorithm_configs(
                policy_config,
                safe_policy_specific_params[i],
                _allow_unknown_configs=True,
            )
        policy_config.rollouts(num_rollout_workers=0)
        policy_config.offline_data(input_="sampler")
        policy_config.evaluation(evaluation_num_workers=0)
        policy_config = policy_config.update_from_dict({"__policy_id": policy_name})
        policies[policy_name] = PolicySpec(
            policy_class=ORPOPolicy,
            config=policy_config,
        )

    policies["current"] = PolicySpec(
        policy_class=None,
        config=None,
    )

    num_rollout_workers: int = config.num_rollout_workers
    if num_safe_policies >= 1:
        workers_per_safe_policy = (
            percent_safe_policy * num_rollout_workers
        ) / num_safe_policies
        if workers_per_safe_policy < 1:
            raise ValueError(
                "Too few workers for the number of safe policies. "
                "Increase num_rollout_workers/percent_safe_policy or decrease num_safe_policies."
            )
        if workers_per_safe_policy != int(workers_per_safe_policy):
            _log.warning(
                "Safe policies are not evenly divided among workers "
                f"({workers_per_safe_policy:.1f} workers per safe policy)."
            )

    def policy_mapping_fn(
        agent_id,
        episode,
        worker: RolloutWorker,
        percent_safe_policy=percent_safe_policy,
        safe_policy_ids=safe_policy_ids,
        num_rollout_workers=num_rollout_workers,
        **kwargs,
    ):
        worker_index = worker.worker_index - 1
        if (
            num_safe_policies >= 1
            and worker_index < num_rollout_workers * percent_safe_policy
        ):
            safe_policy_index = worker_index % num_safe_policies
            return safe_policy_ids[safe_policy_index]
        return "current"

    policies_to_train = safe_policy_ids + ["current"]

    return policies, policy_mapping_fn, policies_to_train


@ex.automain
def main(
    config,
    log_dir,
    ray_init_kwargs,
    experiment_name,
    AlgorithmClass: Type[Algorithm],
    exp_algo,
    num_training_iters,
    save_freq,
    checkpoint_to_load_policies: Optional[List[str]],
    policy_ids_to_load: Optional[List[str]],
    policy_id_to_load_current_policy: Optional[str],
    checkpoint_to_load_current_policy: Optional[str],
    checkpoint_path: Optional[str],
    num_cpus: int,
    _log: Logger,
):
    temp_dir = tempfile.mkdtemp()
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    ray.init(
        num_cpus=num_cpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=temp_dir,
        **ray_init_kwargs,
    )
    algorithm = AlgorithmClass(
        config,
        logger_creator=build_logger_creator(log_dir, experiment_name),
    )

    if checkpoint_to_load_policies is not None:
        for i in range(len(checkpoint_to_load_policies)):
            if exp_algo == "ORPO":
                safe_policy_ids = config.safe_policy_ids
                policy_map = lambda policy_id: safe_policy_ids[i]
            else:
                policy_map = lambda policy_id: DEFAULT_POLICY_ID
            _log.info(
                f"loading policy {policy_map(None)} from {checkpoint_to_load_policies[i]}..."
            )
            load_policies_from_checkpoint(
                checkpoint_to_load_policies[i],
                algorithm,
                policy_ids=(
                    policy_ids_to_load[i] if policy_ids_to_load is not None else None
                ),
                policy_mapping_fn=policy_map,
            )
    if checkpoint_to_load_current_policy is not None:
        _log.info(f"loading policy current from {checkpoint_to_load_current_policy}...")
        load_policies_from_checkpoint(
            checkpoint_to_load_current_policy,
            algorithm,
            policy_ids=policy_id_to_load_current_policy,
            policy_mapping_fn=lambda policy_id: (
                "current" if exp_algo == "ORPO" else DEFAULT_POLICY_ID
            ),
        )

    if checkpoint_path is not None:
        _log.info(f"Restoring checkpoint at {checkpoint_path}")
        algorithm.restore(checkpoint_path)

    result = None
    for _ in range(num_training_iters):
        _log.info(f"Starting training iteration {algorithm.iteration}")
        result = algorithm.train()

        if algorithm.iteration % save_freq == 0:
            checkpoint = algorithm.save()
            # convert_to_msgpack_checkpoint(checkpoint, algorithm.logdir, AlgorithmClass)
            _log.info(f"Saved checkpoint to {checkpoint}")
    checkpoint = algorithm.save()
    # convert_to_msgpack_checkpoint(checkpoint, algorithm.logdir, AlgorithmClass)
    _log.info(f"Saved final checkpoint to {checkpoint}")

    algorithm.stop()

    return result
