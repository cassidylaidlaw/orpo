from occupancy_measures.envs.tomato_environment import (
    Tomato_Environment,
    create_simple_example,
)


def test_tomato_env(tmp_path):
    for level in range(4):
        filename, _ = create_simple_example(tmp_path)
        env = Tomato_Environment(
            {
                "filepath": filename,
                "horizon": 100,
                "reward_fun": "true",
            }
        )
        env.reset()
        for t in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        assert terminated or truncated
