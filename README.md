# Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking

This repository contains code for the paper "[Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking](https://arxiv.org/abs/2403.03185)". 

ORPO is currently supported for RLHF and four non-RLHF environments. The code for non-RLHF experiments is in this repository, while the RLHF experiment code is at https://github.com/cassidylaidlaw/llm_optimization.


All Python code is under the `occupancy_measures` package. Run

    pip install -r requirements.txt

to install dependencies.

## Training the ORPO policies
Checkpoints for the behavioral cloning (BC) trained base policies are stored within the `data/base_policy_checkpoints` directory. For now, these checkpoints were generated in Python 3.9, but in the future, we will provide checkpoints that work with all python versions. You can use these checkpoints to train your own ORPO policies using the following commands: 

- state-action occupancy measure regularization:
```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs=['$COEFF']' 'checkpoint_to_load_policies=["'$BC_CHECKPOINT'"]' checkpoint_to_load_current_policy=$BC_CHECKPOINT seed=$SEED experiment_tag=state-action 'om_divergence_type=["'$TYPE'"]'
```
- state occupancy measure regularization:
```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV reward_fun=proxy exp_algo=ORPO 'om_divergence_coeffs=['$COEFF']' use_action_for_disc 'checkpoint_to_load_policies=["'$BC_CHECKPOINT'"]' checkpoint_to_load_current_policy=$BC_CHECKPOINT seed=$SEED experiment_tag=state 'om_divergence_type=["'$TYPE'"]'
```
- action distribution regularization (Note that we set the ```om_divergence_type``` variable to log the OM divergence for these runs):
```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV reward_fun=proxy exp_algo=ORPO action_dist_kl_coeff=$COEFF seed=$SEED 'checkpoint_to_load_policies=["'$BC_CHECKPOINT'"]' checkpoint_to_load_current_policy=$BC_CHECKPOINT experiment_tag=AD 'om_divergence_type=["'$TYPE'"]'
```
- true reward:
```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV reward_fun=true exp_algo=ORPO 'om_divergence_coeffs=['$COEFF']' 'checkpoint_to_load_policies=["'$BC_CHECKPOINT'"]' checkpoint_to_load_current_policy=$BC_CHECKPOINT seed=$SEED experiment_tag=state-action
```

You can set ```ENV``` to any of the following options:
- traffic ([repo](https://github.com/shivamsinghal001/flow_reward_misspecification))
- pandemic ([repo](https://github.com/shivamsinghal001/pandemic))
- glucose ([repo](https://github.com/shivamsinghal001/glucose))
- tomato level=4 (defined within ```occupancy_measures/envs/tomato_environment.py```

You can set ```TYPE``` to any of the following divergences:
- $\sqrt{\chi^2}$ divergence: "sqrt_chi2"
- $\chi^2$ divergence: "chi2"
- Kullback-Leibler divergence: "kl"
- Total variation: "tv"
- Wasserstein: "wasserstein"

For our experiments using $\sqrt{\chi^2}$ divergence, we ran experiments with the following range of scale-independent coefficients for each regularization technique: 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01. As per our theory, we multiply these by the standard deviation of the rewards from the base policy in each environment:
- traffic: 2e-4
- pandemic: 0.08
- glucose: 0.05
- tomato: 0.05

For our experiments using KL divergence, we ran experiments with the following range of scale-independent coefficients for each regularization technique: 0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0. These must be multiplied by the per-timestep rewards in each environment:
- traffic: 0.0005
- pandemic: 0.06
- glucose: 0.03
- tomato: 0.8

Various notes:
- To generate the policy without any regularization, simply set ```COEFF``` in the code to 0. These policies will reward hack and can be to replicate our experiments for regularizing away from reward hacking behaviors.
- To run the regularizing away experiments, you must add the following variable definition to the commands above in addition to negating the coefficients that you use: ```policy_ids_to_load='[["current"]]'```
- If you do not wish to initialize the policies using the base policy, simply remove ```checkpoint_to_load_current_policy``` from the commands above. This is needed for replicating our tomato environment results as we start from a randomly initialized policy.
- SUMO is a dependency of the traffic environment, but to run all experiments, you will need to set the ```SUMO_HOME``` environment variable. This requires you to first install SUMO, which can generally be done using the following command ```apt install sumo sumo-tools sumo-doc```. Please refer to the traffic environment repository for more information. If you would like to run the experiments without installing this dependency, feel free to comment out any references to the traffic environment within the main package.

## Working with the base policies

In order to generate your own base policies for each of the pandemic, glucose, and traffic environments, you can run the following commands:
1. Generate a dummy checkpoint which will be evaluated to generate rollouts:

```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV exp_algo=SafePolicyGenerationAlgorithm num_rollout_workers=10 safe_policy_action_dist_input_info_key=$ACTION_INFO_KEY num_training_iters=0
```
You must set ```safe_policy_action_dist_input_info_key``` differently for each of the environments.
- For the traffic environment, the key is "acc_controller_actions".
- For the glucose environment, the key is "glucose_pid_controller".
- For the pandemic environment, we generated checkpoints for each of the following keys, and combined the data: "S0", "S1", "S2", "S3", "S4", "S0-4-0", "S0-4-0-FI", "S0-4-0-GI", "swedish_strategy", "italian_strategy". 

2. Now, evaluate the checkpoint that you just generated to get a dataset:
  
```
python -m occupancy_measures.experiments.evaluate with run=SafePolicyGenerationAlgorithm episodes=1000 "policy_ids=['safe_policy0']" num_workers=20 experiment_name=1000-episodes checkpoint=$CHECKPOINT
```
3. Now, you must train a BC policy using the data you just generated. You might get a better policy if there is added entropy. Use the following command:

```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=$ENV exp_algo=BC save_freq=5 evaluation_num_workers=10 evaluation_interval=5 evaluation_duration=20 input=$INPUT entropy_coeff=$COEFF
```

For the tomato environment, you must just train a PPO agent using the following command:

```
python -m occupancy_measures.experiments.orpo_experiments with env_to_run=tomato level=4 reward_fun=true save_freq=1 randomness_eps=0.1
```

## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@inproceedings{laidlaw2023rewardhacking,
  title={Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking},
  author={Laidlaw, Cassidy and Singhal, Shivam and Dragan, Anca},
  booktitle={arXiv preprint},
  year={2023}
}
```
