import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.utils.typing import AlgorithmConfigDict

from ..envs.glucose_callbacks import GlucoseCallbacks
from ..models.glucose_models import GlucoseModelConfig
from ..models.model_with_discriminator import ModelWithDiscriminatorConfig
from ..utils.training_utils import convert_sacred_dict


class GlucoseBeta(TorchBeta):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("low", 0.0)
        kwargs.setdefault("high", 0.1)
        super().__init__(*args, **kwargs)


ModelCatalog.register_custom_action_dist("glucose_beta", GlucoseBeta)


def create_glucose_config(ex):
    @ex.config
    def run_glucose(env_to_run, _log):
        if env_to_run == "glucose":
            # Environment
            env_name = "glucose_env"
            proxy_reward_fun = "expected_patient_cost"
            true_reward_fun = "magni_bg"
            reward_fun = "proxy"
            patient_name = "adult#001"
            seeds = None  # default: {"numpy": 0, "sensor": 0, "scenario": 0}
            reset_lim = {"lower_lim": 10, "upper_lim": 1000}
            time = False
            meal = False
            bw_meals = True
            load = False
            use_pid_load = False
            hist_init = True
            gt = False
            n_hours = 4
            norm = False
            time_std = None
            use_old_patient_env = False
            action_cap = None
            action_bias = 0
            action_scale = "basal"
            basal_scaling = 43.2
            meal_announce = None
            residual_basal = False
            residual_bolus = False
            residual_PID = False
            fake_gt = False
            fake_real = False
            suppress_carbs = False
            limited_gt = False
            weekly = False
            update_seed_on_reset = True
            deterministic_meal_size = False
            deterministic_meal_time = False
            deterministic_meal_occurrence = False
            harrison_benedict = True
            restricted_carb = False
            meal_duration = 5
            rolling_insulin_lim = None
            universal = False
            reward_bias = 0
            carb_error_std = 0
            carb_miss_prob = 0
            source_dir = ""
            noise_scale = 0
            model = None
            model_device = "cuda" if torch.cuda.is_available() else "cpu"
            use_model = False
            unrealistic = False
            use_custom_meal = False
            custom_meal_num = 3
            custom_meal_size = 1
            start_date = None
            use_only_during_day = False
            horizon_days = 20
            horizon = horizon_days * 12 * 24
            reward_scale = 1e-3 if reward_fun == "true" else 1
            termination_penalty = 1e2 / reward_scale
            use_safe_policy_actions = False
            safe_policy_noise_std_dev = 0.003

            env_config = {
                "proxy_reward_fun": proxy_reward_fun,
                "true_reward_fun": true_reward_fun,
                "reward_fun": reward_fun,
                "patient_name": patient_name,
                "seeds": seeds,
                "reset_lim": reset_lim,
                "time": time,
                "meal": meal,
                "bw_meals": bw_meals,
                "load": load,
                "use_pid_load": use_pid_load,
                "hist_init": hist_init,
                "gt": gt,
                "n_hours": n_hours,
                "norm": norm,
                "time_std": time_std,
                "use_old_patient_env": use_old_patient_env,
                "action_cap": action_cap,
                "action_bias": action_bias,
                "action_scale": action_scale,
                "basal_scaling": basal_scaling,
                "meal_announce": meal_announce,
                "residual_basal": residual_basal,
                "residual_bolus": residual_bolus,
                "residual_PID": residual_PID,
                "fake_gt": fake_gt,
                "fake_real": fake_real,
                "suppress_carbs": suppress_carbs,
                "limited_gt": limited_gt,
                "termination_penalty": termination_penalty,
                "weekly": weekly,
                "update_seed_on_reset": update_seed_on_reset,
                "deterministic_meal_size": deterministic_meal_size,
                "deterministic_meal_time": deterministic_meal_time,
                "deterministic_meal_occurrence": deterministic_meal_occurrence,
                "harrison_benedict": harrison_benedict,
                "restricted_carb": restricted_carb,
                "meal_duration": meal_duration,
                "rolling_insulin_lim": rolling_insulin_lim,
                "universal": universal,
                "reward_bias": reward_bias,
                "carb_error_std": carb_error_std,
                "carb_miss_prob": carb_miss_prob,
                "source_dir": source_dir,
                "model": model,
                "model_device": model_device,
                "use_model": use_model,
                "unrealistic": unrealistic,
                "noise_scale": noise_scale,
                "use_custom_meal": use_custom_meal,
                "custom_meal_num": custom_meal_num,
                "custom_meal_size": custom_meal_size,
                "start_date": start_date,
                "horizon": horizon,
                "use_only_during_day": use_only_during_day,
                "reward_scale": reward_scale,
                "use_safe_policy_actions": use_safe_policy_actions,
                "safe_policy_noise_std_dev": safe_policy_noise_std_dev,
            }

            callbacks = GlucoseCallbacks

            # Training
            num_rollout_workers = 10
            num_envs_per_worker = 5
            seed = 0
            num_gpus = 1 if torch.cuda.is_available() else 0
            num_training_iters = 500  # noqa: F841
            lr = 1e-4
            grad_clip = 10
            gamma = 0.99
            use_gae = True
            gae_lambda = 0.98
            vf_loss_coeff = 1e-4
            vf_clip_param = 100
            entropy_coeff = 0.01
            entropy_coeff_start = entropy_coeff
            entropy_coeff_end = entropy_coeff
            entropy_coeff_horizon = 1e6
            entropy_coeff_schedule = [
                [0, entropy_coeff_start],
                [entropy_coeff_horizon, entropy_coeff_end],
            ]
            rollout_fragment_length = 2000  # default: horizon
            train_batch_size = max(
                rollout_fragment_length * num_rollout_workers * num_envs_per_worker,
                rollout_fragment_length * num_envs_per_worker,
            )
            sgd_minibatch_size = min(1024, train_batch_size)
            kl_coeff = 0.2
            kl_target = 1e-3
            clip_param = 0.05
            num_sgd_iter = 4
            batch_mode = "truncate_episodes"

            # Model
            vf_share_layers = True
            num_layers = 3
            hidden_size = 64
            discriminator_width = 256
            discriminator_depth = 2
            discriminator_state_dim = 2
            use_cgm_for_obs = True
            use_subcutaneous_glucose_obs = False
            if use_cgm_for_obs or use_subcutaneous_glucose_obs:
                discriminator_state_dim = 1
            use_action_for_disc = True
            use_history_for_disc = False
            time_dim = 2
            disc_history = 48  # how many intervals of five minutes of history to use for discriminator
            if not use_history_for_disc:
                disc_history = 1
            history_range = (disc_history * -1, 0)
            model_action_scale = 10
            glucose_custom_model_config: GlucoseModelConfig = {
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "action_scale": model_action_scale,
                "use_cgm_for_obs": use_cgm_for_obs,
                "use_subcutaneous_glucose_obs": use_subcutaneous_glucose_obs,
            }
            custom_model_config: ModelWithDiscriminatorConfig = {
                "discriminator_depth": discriminator_depth,
                "discriminator_width": discriminator_width,
                "discriminator_state_dim": discriminator_state_dim,
                "use_action_for_disc": use_action_for_disc,
                "use_history_for_disc": use_history_for_disc,
                "time_dim": time_dim,
                "history_range": history_range,
                "env_specific_config": glucose_custom_model_config,
            }

            model_config = {
                "custom_model": "glucose",
                "custom_model_config": custom_model_config,
                "vf_share_layers": vf_share_layers,
                "custom_action_dist": "GlucoseBeta",
            }

            config = AlgorithmConfig().rl_module(_enable_rl_module_api=False)
            config_updates: AlgorithmConfigDict = {  # noqa: F841
                "env": env_name,
                "env_config": convert_sacred_dict(env_config),
                "callbacks": callbacks,
                "num_rollout_workers": num_rollout_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "train_batch_size": train_batch_size,
                "sgd_minibatch_size": sgd_minibatch_size,
                "batch_mode": batch_mode,
                "num_sgd_iter": num_sgd_iter,
                "lr": lr,
                "grad_clip": grad_clip,
                "gamma": gamma,
                "use_gae": use_gae,
                "lambda": gae_lambda,
                "vf_loss_coeff": vf_loss_coeff,
                "vf_clip_param": vf_clip_param,
                "entropy_coeff_schedule": entropy_coeff_schedule,
                "kl_coeff": kl_coeff,
                "kl_target": kl_target,
                "clip_param": clip_param,
                "num_gpus": num_gpus,
                "rollout_fragment_length": rollout_fragment_length,
                "model": model_config,
                "framework_str": "torch",
                "normalize_actions": False,
                "seed": seed,
            }
            config.update_from_dict(config_updates)
