# import os
# import time
# import signal
# import matplotlib.pyplot as plt
import gymnasium as gymn
from gymnasium import spaces
# from gymnasium import error, utils
from gymnasium.utils import seeding
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

TB = 1024 * 1024 * 1024  # filesize is in kb so here TB is this.


class CacheEnv(gymn.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    actions_num = 1  # best guess if the file is in the cache/should be kept in cache

    def __init__(self, InputData, CacheSize, render_mode='human'):
        self.render_mode = render_mode
        self.name = '{}TB'.format(CacheSize/TB)
        self.actor_name = 'default'

        self.accesses_filename = InputData + '.pa'

        self.load_access_data()
      #  self.seed()  # probably not needed

        self.cache_value_weight = 1.0  # applies only on files already in cache

        self.cache_size = CacheSize
        self.cache_hwm = .95 * self.cache_size
        self.cache_lwm = .90 * self.cache_size
        self.cache_kbytes = 0
        # tuples of fid: [access_no, filesize, decission]
        self.cache_content = {}
        self.files_processed = 0
        self.data_processed = 0

        self.monitoring = []

        # from previous cycle
        self.weight = 0  # delivered in next cycle.
        self.found_in_cache = False
        self.fID = None
        #########

        self.viewer = None

        maxes = self.accesses.max()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            # first 6 are tokens, 7th is filesize, 8th is how full is cache at the moment
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([maxes.iloc[0], maxes.iloc[1], maxes.iloc[2], maxes.iloc[3],
                           maxes.iloc[4], maxes.iloc[5], maxes.iloc[6], 100]),
            dtype=np.int64
        )
        print('environment loaded!  cache size [kB]:', self.cache_size)

    def set_actor_name(self, actor):
        self.actor_name = actor

    def load_access_data(self):
        # last variable is the fileID.
        self.accesses = pd.read_parquet(self.accesses_filename)
        # self.accesses = self.accesses.head(50000)
        self.total_accesses = self.accesses.shape[0]
        print("accesses loaded:", self.total_accesses)

    def save_monitoring_data(self):
        mdata = pd.DataFrame(self.monitoring, columns=[
                             'kB', 'cache size', 'cache hit', 'reward'])
        mdata.to_parquet('results/' + self.name + '_' +
                         self.actor_name + '.pa', engine='pyarrow')


    def _cache_cleanup(self):
        # create dataframe from cache info
        acc = pd.DataFrame.from_dict(self.cache_content, orient='index', columns=[
                                     'accNo', 'fs', 'action'])
        # move accesses that we want kept to much later times
        acc.loc[:,'accNo'] = acc['accNo'] + acc['action']*1000000
        # order files by access instance (equivalent of time)
        acc.sort_values(by='accNo', axis=0, inplace=True)
        # starting from lowest number remove from cache
        counter = 0
        while self.cache_kbytes > self.cache_lwm:
            row = acc.iloc[counter, :]
            self.cache_kbytes -= row['fs']
            del self.cache_content[row.name]
            counter += 1
        # print('cleaned', counter, 'files.')

    def step(self, action):

        # calculate reward, this is for the access delivered in previous state
        # takes into account:
        #  * weight (was remembered from previous step),
        #  * was it in cache (was remembered from previous step),
        #  * what action actor took
        reward = self.weight
        if (self.found_in_cache and action == 0) or (not self.found_in_cache and action == 1):
            reward = -reward

        if self.fID in self.cache_content:  # check that this is not the first access
            # remember what is the action
            prevCacheContent = self.cache_content[self.fID]
            self.cache_content[self.fID] = (
                prevCacheContent[0], prevCacheContent[1], action)

            # checks if cache hit HWM
            # print('cache filled:', self.cache_kbytes)
            if self.cache_kbytes > self.cache_hwm:
                # print('cache cleanup starting on access:', self.files_processed)
                self._cache_cleanup()

        # takes next access
        row = self.accesses.iloc[self.files_processed, :]
        self.fID = row['fID']
        fs = row['kB']
        # print(row['1'], row['2'], row['3'], row['4'], row['5'], row['6'], row['kB'], row['fID'])

        self.found_in_cache = self.fID in self.cache_content
        # print('found in cache', self.found_in_cache, self.fID, self.cache_content)
        if self.found_in_cache:
            # print('cache hit - 5%')
            self.weight = fs * self.cache_value_weight
        else:
            # print('cache miss - 100%')
            self.weight = fs
            self.cache_kbytes += fs

        self.cache_content[self.fID] = (self.files_processed, fs, )

        self.monitoring.append(
            [fs, self.cache_kbytes, self.found_in_cache, int(reward)])

        self.files_processed += 1
        self.data_processed += fs

        state = [row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],
                 fs, self.cache_kbytes * 100 // self.cache_size]

        # Add terminated and truncated flags
        terminated = False
        truncated = False    
        if self.files_processed >= self.total_accesses:
            self.save_monitoring_data()
            terminated = True
        return np.array(state), int(reward), terminated, truncated, {}
    
    def reset(self, seed=None, options=None):

        #Add seed and options arguments
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        row = self.accesses.iloc[0]
        self.files_processed = 0
        self.cache_content = {}
        self.cache_kbytes = 0
        self.monitoring = []
        self.fID = row['fID']
        self.found_in_cache = False
        self.weight = 0
        fs = row['kB']
       # print('dtype for observation space:', self.observation_space.dtype)
        
        obs = np.array([row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],
                 fs, self.cache_kbytes * 100 // self.cache_size])
      #  print('dtype for obs ', obs.dtype)
        return obs, {}

    def render(self, mode='human'):
        # screen_width = 600
        # screen_height = 400
        # if self.viewer is None:  # creation of entities.
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -20 / 2, 20 / 2, 40 / 2, -40 / 2
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return

    def close(self):
        if len(self.monitoring) > 0:
            self.save_monitoring_data()
        if self.viewer:
            self.viewer.close()
            self.viewer = None
