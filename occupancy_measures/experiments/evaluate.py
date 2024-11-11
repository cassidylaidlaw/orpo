import faulthandler
import json
import logging
import os
import signal
import tempfile
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import PolicyID
from ray.tune.registry import register_env
from sacred import SETTINGS, Experiment

from occupancy_measures.models.model_with_discriminator import ModelWithDiscriminator

from ..agents.orpo import ORPO, chi2_divergence
from ..utils.os_utils import available_cpu_count
from ..utils.training_utils import load_algorithm, load_algorithm_config

os.environ["DISPLAY"] = ":99"

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
CURRENT_POLICY_ID = "current"
SAFE_POLICY_ID = "safe_policy0"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


faulthandler.register(signal.SIGUSR1)


ex = Experiment("traffic_eval", save_git_info=False)
logger = logging.getLogger(__name__)


@ex.config
def sacred_config(_log):
    num_cpus = available_cpu_count()
    config_updates = {}
    run = "PPO"  # noqa: F841
    generate_histogram = False  # noqa: F841
    checkpoint = ""  # noqa: F841
    episodes = 1  # noqa: F841
    evaluation_duration_unit = "episodes"  # noqa: F841
    experiment_name = ""  # noqa: F841
    policy_ids: list = [CURRENT_POLICY_ID]  # noqa: F841
    hist_x_label: str = ""  # noqa: F841
    seed = 17
    render = (
        False  # only possible for the traffic and tomato environments at the moment
    )
    render_dir_name = "tomato_rendering"

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(  # noqa: F841
        os.path.dirname(checkpoint), f"rollouts_{experiment_name}_{time_str}"
    )

    # For traffic environment eval, run with xvfb-run if running on a server
    original_config = load_algorithm_config(checkpoint)
    if original_config["env_config"].get("flow_params") is not None:
        from flow.utils.registry import make_create_env
        from flow.utils.rllib import FlowParamsEncoder, get_flow_params

        original_flow_params = get_flow_params(original_config.__dict__)
        original_reward_spec = original_config["env_config"]["reward_specification"]
        original_reward_fun = original_config["env_config"]["reward_fun"]

        if render:
            original_sim = original_flow_params["sim"]
            render_mode = "drgb"
            original_sim.render = render_mode
            original_sim.restart_instance = True
            original_sim.save_render = True

            original_sim.force_color_updates = True

            original_flow_params["sim"] = original_sim

        flow_json = json.dumps(
            original_flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
        )

        create_env, env_name = make_create_env(
            params=original_flow_params,
            reward_specification=original_reward_spec,
            reward_fun=original_reward_fun,
            path=out_dir,
        )
        register_env(env_name, make_multi_agent(create_env))
        config_updates["env"] = env_name
        config_updates["env_config"] = {"flow_params": flow_json}
    elif "tomato" in original_config["env"] and render:
        config_updates["env_config"] = original_config["env_config"]
        config_updates["env_config"]["rendering_filepath"] = os.path.join(
            os.path.dirname(checkpoint), f"{render_dir_name}_{time_str}"
        )

        config_updates["env_config"]["render_mode"] = "rgb_array"
    is_multiagent = False
    per_policy_config_updates: Dict[PolicyID, Any] = {
        policy_id: {} for policy_id in policy_ids
    }
    if is_multiagent:
        for policy_id in policy_ids:
            per_policy_config_updates[policy_id].setdefault("multiagent", {})
            policy_mapping_fn = (
                lambda agent_id, *args, policy_id=policy_id, **kwargs: policy_id
            )
            per_policy_config_updates[policy_id]["multiagent"][
                "policy_mapping_fn"
            ] = policy_mapping_fn
            per_policy_config_updates[policy_id][
                "policy_mapping_fn"
            ] = policy_mapping_fn
            per_policy_config_updates[policy_id]["policies_to_train"] = [policy_id]
    num_workers = 1
    # Could use dataset
    train_batch_size = original_config["train_batch_size"]
    num_envs_per_worker = original_config["num_envs_per_worker"]
    rollout_fragment_length = original_config["rollout_fragment_length"]

    updates = {  # noqa: F841
        "_enable_rl_module_api": False,
        "_enable_learner_api": False,
        "enable_connectors": False,
        "seed": seed,
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_duration": (
            episodes * original_config["env_config"].get("horizon", 1)
            if evaluation_duration_unit == "timesteps"
            else episodes
        ),
        "evaluation_duration_unit": evaluation_duration_unit,
        "evaluation_config": {},
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "disable_env_checking": True,
        "evaluation_sample_timeout_s": 300,
        "output": out_dir,
        "train_batch_size": train_batch_size,
        "num_rollout_workers": 0,
        "num_envs_per_worker": num_envs_per_worker,
        "rollout_fragment_length": rollout_fragment_length,
    }

    # for PPO and ORPO policies
    if "num_sgd_iter" in original_config:
        updates["num_sgd_iter"] = original_config["num_sgd_iter"]
    if "sgd_minibatch_size" in original_config:
        updates["sgd_minibatch_size"] = original_config["sgd_minibatch_size"]

    config_updates.update(updates)

    for policy_id in policy_ids:
        per_policy_config_updates[policy_id].update(
            {
                **config_updates,
                "output": f"{out_dir}_{policy_id}",
            }
        )

    extra_config_updates = {}  # noqa: F841

    use_local_mode = False
    ray_init_kwargs = {"local_mode": use_local_mode, "num_cpus": num_cpus}  # noqa: F841


@ex.automain
def main(
    run: str,
    episodes: int,
    per_policy_config_updates: dict,
    extra_config_updates: dict,
    policy_ids: list,
    checkpoint: str,
    out_dir: str,
    generate_histogram: bool,
    hist_x_label: str,
    ray_init_kwargs: dict,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(),
        **ray_init_kwargs,
    )

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_results: Dict[str, Any] = {}
    model = None
    current_policy_action_dist_class = None
    safe_policy_action_dist_class = None
    safe_policy = None

    # to make sure safe policies come first when setting the model
    if CURRENT_POLICY_ID in policy_ids:
        policy_ids.remove(CURRENT_POLICY_ID)
        policy_ids.sort()
        policy_ids.append(CURRENT_POLICY_ID)

    for policy_id in policy_ids:
        config_updates = per_policy_config_updates[policy_id]
        config_updates = Algorithm.merge_algorithm_configs(
            config_updates, extra_config_updates, _allow_unknown_configs=True
        )
        algorithm = load_algorithm(checkpoint, run, config_updates)
        policy = algorithm.get_policy(policy_id)
        assert isinstance(policy, TorchPolicyV2)
        assert algorithm.evaluation_workers is not None
        selected_eval_worker_ids = [
            worker_id
            for i, worker_id in enumerate(
                algorithm.evaluation_workers.healthy_worker_ids()
            )
            if i * 1 < episodes
        ]
        assert isinstance(algorithm.config, AlgorithmConfig)
        batches = algorithm.evaluation_workers.foreach_worker(
            func=lambda w: w.sample(),
            local_worker=False,
            remote_worker_ids=selected_eval_worker_ids,
            timeout_seconds=algorithm.config.evaluation_sample_timeout_s,
        )
        eval_results[policy_id] = algorithm.evaluate()["evaluation"]
        if CURRENT_POLICY_ID in policy_id:
            current_policy_action_dist_class = policy.dist_class

        if (len(policy_ids) > 1 and CURRENT_POLICY_ID not in policy_id) or len(
            policy_ids
        ) == 1:
            model = policy.model
            assert isinstance(model, TorchModelV2)
            model.to(device)
            safe_policy_action_dist_class = policy.dist_class
            safe_policy = policy
            if CURRENT_POLICY_ID in policy_id:
                if algorithm.get_policy(SAFE_POLICY_ID) is not None:
                    safe_policy = algorithm.get_policy(SAFE_POLICY_ID)
                    safe_policy_action_dist_class = safe_policy.dist_class
                    model = safe_policy.model
                    logger.info("Loaded safe policy from algorithm successfully!")
                else:
                    logger.warn(
                        "Using untrained current policy discriminator for generating discriminator scores!"
                    )

        occupancy_measure_kl_vals = []
        occupancy_measure_chi2_vals = []
        occupancy_measure_sqrt_chi2_vals = []

        action_distribution_kl_vals = []
        action_distribution_chi2_vals = []
        action_distribution_sqrt_chi2_vals = []

        data = []
        if isinstance(algorithm, ORPO):
            for file_data in batches:
                file_data = file_data.as_multi_agent()
                file_data_cpu = file_data.policy_batches[policy_id].copy()
                file_data_cuda = file_data.policy_batches[policy_id].to_device(device)
                data.append(file_data_cuda)
                if CURRENT_POLICY_ID in policy_id:
                    discriminator_policy_scores = model.discriminator(file_data_cuda)
                    # kl
                    occupancy_measure_kl = np.mean(
                        discriminator_policy_scores[:, 0].cpu().detach().numpy()
                    )
                    occupancy_measure_kl_vals.append(occupancy_measure_kl)

                    # chi2
                    occupancy_measure_chi2_scores = (
                        discriminator_policy_scores[:, 0].exp().cpu().detach().numpy()
                        - 1
                    )
                    occupancy_measure_chi2 = np.mean(occupancy_measure_chi2_scores)
                    occupancy_measure_chi2_vals.append(occupancy_measure_chi2)

                    # sqrt chi2
                    occupancy_measure_chi2_sorted = np.sort(
                        occupancy_measure_chi2_scores
                    )
                    i = int(0.01 * len(occupancy_measure_chi2_sorted))
                    occupancy_measure_chi2_trimmed = np.mean(
                        occupancy_measure_chi2_sorted[i:-i]
                    )
                    if occupancy_measure_chi2_trimmed <= 0:
                        occupancy_measure_sqrt_chi2_scores = (
                            occupancy_measure_chi2_scores
                        )
                    else:
                        occupancy_measure_sqrt_chi2_scores = (
                            occupancy_measure_chi2_scores
                            / np.sqrt(occupancy_measure_chi2_trimmed)
                        )
                    occupancy_measure_sqrt_chi2 = np.mean(
                        occupancy_measure_sqrt_chi2_scores
                    )
                    occupancy_measure_sqrt_chi2_vals.append(occupancy_measure_sqrt_chi2)

                    safe_policy_action_dist_inputs = (
                        algorithm.get_safe_policy_dist_inputs(
                            current_batch=file_data_cpu,
                            safe_policy_model=(
                                model.to("cpu")
                                if ray_init_kwargs["local_mode"]
                                else model
                            ),
                            safe_policy=safe_policy,
                        )
                    )
                    if safe_policy_action_dist_class is not None:
                        safe_policy_action_dist = safe_policy_action_dist_class(
                            torch.from_numpy(safe_policy_action_dist_inputs)
                            .cpu()
                            .detach(),
                            None,
                        )
                        current_action_dist = current_policy_action_dist_class(
                            file_data_cuda[SampleBatch.ACTION_DIST_INPUTS]
                            .cpu()
                            .detach(),
                            None,
                        )

                        # AD KL
                        action_distribution_kl = current_action_dist.kl(
                            safe_policy_action_dist
                        ).mean()

                        # AD chi2
                        action_distribution_chi2_scores = chi2_divergence(
                            current_action_dist, safe_policy_action_dist
                        )
                        action_distribution_chi2_scores = torch.where(
                            torch.isfinite(action_distribution_chi2_scores),
                            action_distribution_chi2_scores,
                            action_distribution_kl,
                        )
                        action_distribution_chi2 = torch.mean(
                            action_distribution_chi2_scores
                        )

                        # AD sqrt chi2
                        action_distribution_sqrt_chi2_scores = torch.sqrt(
                            action_distribution_chi2_scores.clamp(min=1e-4)
                        )
                        action_distribution_sqrt_chi2 = torch.mean(
                            action_distribution_sqrt_chi2_scores
                        )
                    else:
                        logger.warn(
                            "Unable to calculate action distribution divergence without safe policy action distribution class specified!"
                        )
                        action_distribution_kl = torch.tensor(0)
                        action_distribution_chi2 = torch.tensor(0)
                        action_distribution_sqrt_chi2 = torch.tensor(0)
                    action_distribution_kl_vals.append(action_distribution_kl.item())
                    action_distribution_chi2_vals.append(
                        action_distribution_chi2.item()
                    )
                    action_distribution_sqrt_chi2_vals.append(
                        action_distribution_sqrt_chi2.item()
                    )
                else:
                    logger.warn(
                        "OM and AD divergences can't be calculated for safe policies"
                    )
                    occupancy_measure_kl_vals.append(0)
                    occupancy_measure_chi2_vals.append(0)
                    action_distribution_kl_vals.append(0)
                    action_distribution_chi2_vals.append(0)
                    action_distribution_sqrt_chi2_vals.append(0)

        vals_of_interest = {
            "mean_true_reward": eval_results[policy_id]["custom_metrics"][
                "true_reward_mean"
            ],
            "proxy_reward_mean": eval_results[policy_id]["custom_metrics"][
                "proxy_reward_mean"
            ],
            "average_action_distribution_kl": np.mean(
                action_distribution_kl_vals, dtype="float64"
            ),
            "action_distribution_kl_vals": np.array(
                action_distribution_kl_vals, dtype="float64"
            ).tolist(),
            "average_occupancy_measure_kl": np.mean(
                occupancy_measure_kl_vals, dtype="float64"
            ),
            "occupancy_measure_kl_vals": np.array(
                occupancy_measure_kl_vals, dtype="float64"
            ).tolist(),
            "average_occupancy_measure_chi2": np.mean(
                occupancy_measure_chi2_vals, dtype="float64"
            ),
            "occupancy_measure_chi2_vals": np.array(
                occupancy_measure_chi2_vals, dtype="float64"
            ).tolist(),
            "average_occupancy_measure_sqrt_chi2": np.mean(
                occupancy_measure_sqrt_chi2_vals, dtype="float64"
            ),
            "occupancy_measure_sqrt_chi2_vals": np.array(
                occupancy_measure_sqrt_chi2_vals, dtype="float64"
            ).tolist(),
            "average_action_distribution_chi2": np.mean(
                action_distribution_chi2_vals, dtype="float64"
            ),
            "action_distribution_chi2_vals": np.array(
                action_distribution_chi2_vals, dtype="float64"
            ).tolist(),
            "average_action_distribution_sqrt_chi2": np.mean(
                action_distribution_sqrt_chi2_vals, dtype="float64"
            ),
            "action_distribution_sqrt_chi2_vals": np.array(
                action_distribution_sqrt_chi2_vals, dtype="float64"
            ).tolist(),
        }
        with open(
            out_dir + "/output_metrics_" + policy_id + ".json", "w"
        ) as vals_of_interest_file:
            json.dump(vals_of_interest, vals_of_interest_file)
            vals_of_interest_file.close()
        with open(out_dir + "/eval_result_" + policy_id + ".json", "w") as out_file:
            json.dump(eval_results[policy_id], out_file, cls=NpEncoder)
            out_file.close()
        episode_lengths = np.sum(
            eval_results[policy_id]["hist_stats"]["episode_lengths"]
        )
        episode_rewards = np.sum(
            eval_results[policy_id]["hist_stats"]["episode_reward"]
        )
        average_reward = episode_rewards / episode_lengths
        logger.info("Saved stats for the policy with ID " + policy_id)
        logger.info("Average Reward per timestep: " + str(average_reward))

        if generate_histogram and isinstance(algorithm, ORPO):
            assert isinstance(model, ModelWithDiscriminator)
            observations: Any = []
            disc_outputs: Any = []
            for eps_data in data:
                observations += model.process_obs(eps_data).flatten(0).tolist()
                disc_outputs += model.discriminator(eps_data).tolist()

            observations, disc_outputs = zip(*sorted(zip(observations, disc_outputs)))

            fig, ax1 = plt.subplots()
            plt.title(policy)
            ax1.set_xlabel(hist_x_label)

            ax1.set_ylabel("Counts", color="darkblue")
            ax1.hist(observations, color="blue")
            ax1.tick_params(axis="y", labelcolor="darkblue")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Discriminator Outputs", color="darkorange")
            ax2.plot(observations, disc_outputs, color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")

            fig.savefig(out_dir + "/" + policy_id + ".png")

        algorithm.stop()

    if eval_results:
        return eval_results
