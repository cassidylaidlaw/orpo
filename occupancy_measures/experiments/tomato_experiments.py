import os

import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import AlgorithmConfigDict

from occupancy_measures.models.model_with_discriminator import (
    ModelWithDiscriminatorConfig,
)

from ..envs.tomato_callbacks import TomatoCallbacks
from ..envs.tomato_environment import create_simple_example


def create_tomato_config(ex):
    @ex.config
    def tomato_config(env_to_run, experiment_parts, _log):
        if env_to_run == "tomato":
            # Environment
            env_name = "tomato_env"
            level = 2
            filepath, diff = create_simple_example("data/", level)
            experiment_parts.append(diff)
            horizon = 100
            reward_fun = "true"
            dry_distance = 3
            reward_factor = 0.02
            neg_rew = -0.1
            randomness_eps = None
            use_noop = False
            render_mode = None
            rendering_filepath = "data/tomato_renderings"
            if not os.path.exists(rendering_filepath):
                os.makedirs(rendering_filepath)
            env_config = {
                "filepath": filepath,
                "horizon": horizon,
                "reward_fun": reward_fun,
                "dry_distance": dry_distance,
                "reward_factor": reward_factor,
                "neg_rew": neg_rew,
                "randomness_eps": randomness_eps,
                "use_noop": use_noop,
                "render_mode": render_mode,
                "rendering_filepath": rendering_filepath,
            }
            callbacks = TomatoCallbacks

            # Training
            num_rollout_workers = 30
            seed = 0
            num_gpus = 1 if torch.cuda.is_available() else 0
            sgd_minibatch_size = 128
            num_training_iters = 500  # noqa: F841
            lr = 1e-3
            grad_clip = 0.1
            gamma = 0.99
            gae_lambda = 0.98
            vf_clip_param = np.inf
            vf_loss_coeff = 1e-2
            entropy_coeff = 0.01
            entropy_coeff_start = entropy_coeff
            entropy_coeff_end = entropy_coeff
            entropy_coeff_horizon = 2e5
            entropy_coeff_schedule = [
                [0, entropy_coeff_start],
                [entropy_coeff_horizon, entropy_coeff_end],
            ]
            rollout_fragment_length = horizon
            train_batch_size = max(
                rollout_fragment_length * num_rollout_workers, rollout_fragment_length
            )
            kl_coeff = 0.2
            kl_target = 0.001
            clip_param = 0.05
            num_sgd_iter = 8

            # model
            width = 512
            depth = 4
            fcnet_hiddens = [width] * depth
            discriminator_width = 256
            discriminator_depth = 2
            use_action_for_disc = True
            vf_share_layers = False
            custom_model_config: ModelWithDiscriminatorConfig = {
                "discriminator_depth": discriminator_depth,
                "discriminator_width": discriminator_width,
                "use_action_for_disc": use_action_for_disc,
            }
            model_config = {
                "custom_model": "model_with_discriminator",
                "fcnet_hiddens": fcnet_hiddens,
                "custom_model_config": custom_model_config,
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
                "lr": lr,
                "grad_clip": grad_clip,
                "gamma": gamma,
                "lambda": gae_lambda,
                "vf_loss_coeff": vf_loss_coeff,
                "vf_clip_param": vf_clip_param,
                "kl_coeff": kl_coeff,
                "kl_target": kl_target,
                "clip_param": clip_param,
                "num_gpus": num_gpus,
                "entropy_coeff_schedule": entropy_coeff_schedule,
                "entropy_coeff": entropy_coeff,
                "rollout_fragment_length": rollout_fragment_length,
                "model": model_config,
                "framework_str": "torch",
                "seed": seed,
            }
            config.update_from_dict(config_updates)
