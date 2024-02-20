import json

import torch
from flow.flow_cfg.get_experiment import get_exp
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.tune.registry import register_env

from occupancy_measures.models.model_with_discriminator import (
    ModelWithDiscriminatorConfig,
)

from ..envs.traffic_callbacks import TrafficCallbacks


def create_traffic_config(ex):
    @ex.config
    def traffic_config(env_to_run, experiment_parts, _log):
        if env_to_run == "traffic":
            # Environment
            exp_tag = "singleagent_merge_bus"  # horizon might need to be updated for bottleneck to 1040
            experiment_parts.append(exp_tag)
            scenario = get_exp(exp_tag)
            flow_params = scenario.flow_params
            flow_json = json.dumps(
                flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
            )
            exp_algo = "PPO"
            reward_fun = "true"
            assert reward_fun in ["true", "proxy"]
            callbacks = TrafficCallbacks
            use_safe_policy_actions = False
            # Rewards and weights
            proxy_rewards = ["vel", "accel", "headway"]
            proxy_weights = [1, 1, 0.1]
            true_rewards = ["commute", "accel", "headway"]
            true_weights = [1, 1, 0.1]
            true_reward_specification = [
                (r, float(w)) for r, w in zip(true_rewards, true_weights)
            ]
            proxy_reward_specification = [
                (r, float(w)) for r, w in zip(proxy_rewards, proxy_weights)
            ]
            reward_specification = {
                "true": true_reward_specification,
                "proxy": proxy_reward_specification,
            }
            reward_scale = 0.0001
            horizon = flow_params["env"].horizon
            flow_params["env"].horizon = horizon
            create_env, env_name = make_create_env(
                params=flow_params,
                reward_specification=reward_specification,
                reward_fun=reward_fun,
                reward_scale=reward_scale,
            )
            register_env(env_name, make_multi_agent(create_env))
            env_config = {
                "flow_params": flow_json,
                "reward_specification": reward_specification,
                "reward_fun": reward_fun,
                "run": exp_algo,
                "use_safe_policy_actions": use_safe_policy_actions,
            }

            # Training
            num_rollout_workers = 10
            seed = 17
            num_gpus = 1 if torch.cuda.is_available() else 0
            rollout_fragment_length = 4000  # default: scenario.N_ROLLOUTS * horizon
            train_batch_size = max(
                num_rollout_workers * rollout_fragment_length, rollout_fragment_length
            )
            sgd_minibatch_size = min(16 * 1024, train_batch_size)
            num_training_iters = 250  # noqa: F841
            lr = 5e-5
            lr_start = lr
            lr_end = lr
            lr_horizon = 1000000
            lr_decay_schedule = [
                [0, lr_start],
                [lr_horizon, lr_end],
            ]
            batch_mode = "truncate_episodes"

            # PPO
            gamma = 0.99
            use_gae = True
            gae_lambda = 0.97
            kl_target = 0.02
            vf_clip_param = 10000
            vf_loss_coeff = 0.5
            grad_clip = None
            num_sgd_iter = 5
            entropy_coeff = 0.01
            entropy_coeff_start = entropy_coeff
            entropy_coeff_end = entropy_coeff
            entropy_coeff_horizon = 1000000
            entropy_coeff_schedule = [
                [0, entropy_coeff_start],
                [entropy_coeff_horizon, entropy_coeff_end],
            ]

            # model
            width = 512
            depth = 4
            fcnet_hiddens = [width] * depth
            discriminator_width = 256
            discriminator_depth = 2
            use_action_for_disc = True
            vf_share_layers = True
            custom_model_config: ModelWithDiscriminatorConfig = {
                "discriminator_depth": discriminator_depth,
                "discriminator_width": discriminator_width,
                "use_action_for_disc": use_action_for_disc,
            }
            model_config = {
                "custom_model": "model_with_discriminator",
                "fcnet_hiddens": fcnet_hiddens,
                "custom_model_config": custom_model_config,
                "custom_action_dist": "TrafficBeta",
                "vf_share_layers": vf_share_layers,
            }

            config = AlgorithmConfig().rl_module(_enable_rl_module_api=False)
            config_updates: AlgorithmConfigDict = {  # noqa: F841
                "env": env_name,
                "env_config": env_config,
                "callbacks": callbacks,
                "num_rollout_workers": num_rollout_workers,
                "train_batch_size": train_batch_size,
                "sgd_minibatch_size": sgd_minibatch_size,
                "num_sgd_iter": num_sgd_iter,
                "gamma": gamma,
                "use_gae": use_gae,
                "lambda": gae_lambda,
                "kl_target": kl_target,
                "vf_clip_param": vf_clip_param,
                "vf_loss_coeff": vf_loss_coeff,
                "grad_clip": grad_clip,
                "lr": lr,
                "lr_schedule": lr_decay_schedule,
                "num_gpus": num_gpus,
                "rollout_fragment_length": rollout_fragment_length,
                "entropy_coeff": entropy_coeff,
                "entropy_coeff_schedule": entropy_coeff_schedule,
                "model": model_config,
                "normalize_actions": False,
                "framework_str": "torch",
                "batch_mode": batch_mode,
                "seed": seed,
            }
            config.update_from_dict(config_updates)
