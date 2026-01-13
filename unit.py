# this actor does always the same action or a random one.

#import gymynasium as gymn
from gym_cache.envs.cache_env import CacheEnv

env = CacheEnv('data/MWT2_processed', 10 * 1024 * 1024 * 1024)
obs, _ = env.reset()

total_reward = 0
for i in range(1000000):
    if not i % 10000:
        print(i, 'total reward', total_reward)
        # env.render()

    # --- random prediction
    # act = env.action_space.sample()
    # --- always predict cache miss
    act = 0

    acc, rew, done, paused, smt = env.step(act)
    # print('access:', acc, 'rew:', rew)
    total_reward += rew

env.close()
print('Finished. Total reward:', total_reward)


# for i in range(10):
#     obs, reward, done, _, _ = env.step(0)
#     print(i, reward, env.weight, env.found_in_cache)
