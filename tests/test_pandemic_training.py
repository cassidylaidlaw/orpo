import glob
import os
import tempfile

import pytest

from occupancy_measures.experiments.orpo_experiments import ex


@pytest.fixture(scope="module")
def default_config():
    log_dir = tempfile.mkdtemp()  # noqa: F841

    return {
        "env_to_run": "pandemic",
        "horizon": 10,
        "train_batch_size": 44,
        "sgd_minibatch_size": 5,
        "num_sgd_iter": 1,
        "num_rollout_workers": 2,
        "num_training_iters": 1,
    }


def test_pandemic_ppo(default_config):
    result = ex.run(
        config_updates={**default_config},
    ).result
    assert result is not None
    assert "true_reward_mean" in result["custom_metrics"]
    assert "proxy_reward_mean" in result["custom_metrics"]
    assert result["training_iteration"] == 1
    assert result["info"]["num_env_steps_trained"] == 44


def test_pandemic_ORPO(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir + "/pandemic/PPO/true/model_128-128/seed_0/*/checkpoint_000001"
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


def test_pandemic_safe_policy_generation(default_config):
    ex.run(
        config_updates={
            **default_config,
            "safe_policy": "S0-4-0",
            "safe_policy_action_dist_input_info_key": "S0-4-0",
            "exp_algo": "SafePolicyGenerationAlgorithm",
            "num_training_iters": 0,
        },
    )
