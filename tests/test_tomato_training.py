import glob
import os
import tempfile

import pytest

from occupancy_measures.envs.tomato_environment import create_simple_example
from occupancy_measures.experiments.orpo_experiments import ex


@pytest.fixture(scope="module")
def default_config():
    log_dir = tempfile.mkdtemp()
    level_file_path, _ = create_simple_example(log_dir, level=2)

    return {
        "log_dir": tempfile.mkdtemp(),
        "num_rollout_workers": 2,
        "num_training_iters": 10,
        "horizon": 10,
        "train_batch_size": 20,
        "sgd_minibatch_size": 5,
        "num_sgd_iter": 1,
        "filepath": level_file_path,
    }


def test_tomato_ppo(default_config):
    result = ex.run(
        config_updates={**default_config},
    ).result
    assert result is not None
    assert "true_reward_mean" in result["custom_metrics"]
    assert "proxy_reward_mean" in result["custom_metrics"]
    assert result["training_iteration"] == 10
    assert result["info"]["num_env_steps_trained"] == 200


def test_tomato_ORPO(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir
        + "/tomato/medium/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000010"
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
    learner_info_base = result["info"]["learner"]["safe_policy0"]
    print(learner_info_base)
    assert "discriminator/curr_policy_score" in learner_info_base["learner_stats"]
    assert "occupancy_measure_kl" in learner_info_base
    assert "action_distribution_kl" in learner_info_base

    # checking for policy KL values
    learner_info_curr = result["info"]["learner"]["current"]
    assert "curr_base_action_distribution_kl" in learner_info_curr["learner_stats"]


def test_tomato_ORPO_tv(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir
        + "/tomato/medium/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000010"
    )
    assert os.path.exists(ppo_checkpoint[0])

    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": ppo_checkpoint,
            "exp_algo": "ORPO",
            "om_divergence_type": ["tv"],
            "percent_safe_policy": 0.5,
        },
    ).result
    assert result is not None
    learner_info = result["info"]["learner"]["safe_policy0"]
    assert "occupancy_measure_tv" in learner_info


def test_tomato_ORPO_was(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir
        + "/tomato/medium/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000010"
    )
    assert os.path.exists(ppo_checkpoint[0])

    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": ppo_checkpoint,
            "exp_algo": "ORPO",
            "om_divergence_type": ["wasserstein"],
            "percent_safe_policy": 0.5,
        },
    ).result
    assert result is not None
    learner_info = result["info"]["learner"]["safe_policy0"]
    assert "occupancy_measure_wasserstein" in learner_info


def test_tomato_ORPO_was_gp(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir
        + "/tomato/medium/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000010"
    )
    assert os.path.exists(ppo_checkpoint[0])

    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": ppo_checkpoint,
            "exp_algo": "ORPO",
            "om_divergence_type": ["wasserstein"],
            "wgan_grad_penalty_weight": 1,
            "percent_safe_policy": 0.5,
        },
    ).result
    assert result is not None
    learner_info = result["info"]["learner"]["safe_policy0"]
    assert "occupancy_measure_wasserstein" in learner_info


def test_tomato_ORPO_safe_policy_confidence(default_config):
    # Execute short PPO run and get the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            "log_dir": checkpoint_dir,
        }
    )
    ppo_checkpoint = glob.glob(
        checkpoint_dir
        + "/tomato/medium/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000010"
    )
    assert os.path.exists(ppo_checkpoint[0])

    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": ppo_checkpoint,
            "exp_algo": "ORPO",
            "om_divergence_type": ["safe_policy_confidence"],
            "percent_safe_policy": 0.5,
        },
    ).result
    assert result is not None
    learner_info = result["info"]["learner"]["safe_policy0"]
    assert "safe_policy_confidence" in learner_info
