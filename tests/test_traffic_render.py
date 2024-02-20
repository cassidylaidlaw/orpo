import glob
import logging
import os
import subprocess
import tempfile

import pytest

from occupancy_measures.experiments.orpo_experiments import ex

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def default_config():
    log_dir = tempfile.mkdtemp()  # noqa: F841
    return {
        "env_to_run": "traffic",
        "num_rollout_workers": 2,
        "horizon": 25,
        "rollout_fragment_length": 25,
        "train_batch_size": 50,
        "sgd_minibatch_size": 20,
        "num_sgd_iter": 1,
        "log_dir": tempfile.mkdtemp(),
        "num_training_iters": 2,
    }


def test_traffic_render(default_config, tmp_path):
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
        + "/traffic/singleagent_merge_bus/PPO/true/model_512-512-512-512/seed_0/*/checkpoint_000002"
    )
    assert os.path.exists(ppo_checkpoint[0])

    command = (
        "xvfb-run python -m occupancy_measures.experiments.evaluate with render=True checkpoint="
        + ppo_checkpoint[0]
        + " out_dir="
        + checkpoint_dir
    )

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logger.info(result.stdout)
    logger.info(result.stderr)
    assert os.path.exists(os.path.join(checkpoint_dir, "flow_rendering"))
