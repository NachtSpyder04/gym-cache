import gymnasium
from gymnasium.utils.env_checker import check_env
# from gym_cache.envs.cache_env import CacheEnv

# env = CustomEnv(arg1, ...)
env = gymnasium.make('gym_cache:Cache-v0')
# env = CacheEnv('data/MWT2_processed', 10 * 1024 * 1024 * 1024)

# It will check your custom environment and output additional warnings if needed
check_env(env.unwrapped)
