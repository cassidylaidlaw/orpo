import glob
import os
import tempfile

import pytest

from occupancy_measures.experiments.orpo_experiments import ex


@pytest.fixture(scope="module")
def default_config():
    log_dir = tempfile.mkdtemp()  # noqa: F841

    return {
        "env_to_run": "glucose",
        "log_dir": tempfile.mkdtemp(),
        "num_rollout_workers": 2,
        "num_training_iters": 2,
        "train_batch_size": 50,
        "num_envs_per_worker": 1,
        "sgd_minibatch_size": 10,
        "rollout_fragment_length": 25,
        "num_sgd_iter": 1,
        "horizon": 25,
        "reset_lim": {"lower_lim": 50, "upper_lim": 200},
    }


def test_glucose_ppo(default_config):
    result = ex.run(
        config_updates=default_config,
    ).result
    assert result is not None
    assert "true_reward_mean" in result["custom_metrics"]
    assert "proxy_reward_mean" in result["custom_metrics"]


def test_glucose_ORPO(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir + "/glucose/PPO/proxy/*/seed_0/*/checkpoint_000002"
    )
    assert os.path.exists(ppo_checkpoint[0])

    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": ppo_checkpoint,
            "exp_algo": "ORPO",
            "percent_safe_policy": 0.5,
        },
    ).result
    assert result is not None
    learner_info = result["info"]["learner"]["safe_policy0"]
    assert "discriminator/curr_policy_score" in learner_info["learner_stats"]
    assert "occupancy_measure_kl" in learner_info


def test_glucose_safe_policy_generation(default_config):
    ex.run(
        config_updates={
            **default_config,
            "safe_policy_action_dist_input_info_key": "glucose_pid_controller",
            "exp_algo": "SafePolicyGenerationAlgorithm",
            "num_training_iters": 0,
        },
    )
