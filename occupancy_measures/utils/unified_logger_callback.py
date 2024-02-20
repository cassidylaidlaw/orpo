from typing import Dict

from ray.tune.experiment.trial import Trial  # noqa: F401
from ray.tune.logger.csv import CSVLoggerCallback
from ray.tune.logger.json import JsonLoggerCallback
from ray.tune.logger.logger import LoggerCallback
from ray.tune.logger.tensorboardx import TBXLoggerCallback


class UnifedLoggerCallback(LoggerCallback):
    def __init__(self):
        self.callbacks = [
            TBXLoggerCallback(),
            CSVLoggerCallback(),
            JsonLoggerCallback(),
        ]

    def log_trial_start(self, trial: "Trial"):
        for callback in self.callbacks:
            callback.log_trial_start(trial)

    def log_trial_restore(self, trial: "Trial"):
        for callback in self.callbacks:
            callback.log_trial_restore(trial)

    def log_trial_save(self, trial: "Trial"):
        for callback in self.callbacks:
            callback.log_trial_save(trial)

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        for callback in self.callbacks:
            callback.log_trial_result(iteration, trial, result)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        for callback in self.callbacks:
            callback.log_trial_end(trial, failed)
