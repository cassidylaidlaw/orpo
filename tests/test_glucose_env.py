import torch
from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv


def test_glucose_env():
    proxy_reward_fun = "magni_bg"
    true_reward_fun = "magni_bg_insulin"
    reward_fun = "true"
    patient_name = "adult#001"
    seed = {"numpy": 0, "sensor": 0, "scenario": 0}
    reset_lim = {"lower_lim": 50, "upper_lim": 200}
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
    termination_penalty = 1e5
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

    env_config = {
        "proxy_reward_fun": proxy_reward_fun,
        "true_reward_fun": true_reward_fun,
        "reward_fun": reward_fun,
        "patient_name": patient_name,
        "seeds": seed,
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
        "use_only_during_day": use_only_during_day,
        "horizon": 10000,
    }

    env = SimglucoseEnv(env_config)
    env.reset()
    terminated = False
    truncated = False
    t = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        t += 1
    assert terminated or truncated
    assert t < 10000
