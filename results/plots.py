
import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
TB = 1024 * 1024 * 1024

# name = 'InfiniteCache_LRU'
# name = '20TB_LRU'
name = '10.0TB_default'


df = pd.read_parquet(name + '.pa')
print("data loaded:", df.shape[0])

print(df)
df['ch_files'] = df['cache hit'].cumsum()
df['CHR files'] = df['ch_files'] / df.index

df['tmp'] = df['cache hit'] * df['kB']
df['ch_data'] = df['tmp'].cumsum()
df['data delivered'] = df['kB'].cumsum()
del df['tmp']
df['CHR data'] = df['ch_data'] / df['data delivered']
df["cache size"] = df["cache size"] / TB
print(df)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.15})

ax22 = ax2.twinx()
f.suptitle(name)
ax1.plot(df["CHR files"], label="CHR (files)")
ax1.plot(df["CHR data"], label="CHR (data)")
ax1.set_ylabel("Cache Hit Ratio")
ax1.legend()
ax1.grid(True)

l1, = ax2.plot(df["reward"].cumsum(), label="Cumulative reward")
l2, = ax22.plot(df["reward"].rolling(5000).mean(), label="Rolling reward")

ax2.set_xlabel("Access Number")
ax2.set_ylabel("Cumulative Reward")
ax22.set_ylabel("Rolling Reward")

ax2.legend(handles=[l1, l2], loc="upper left")
ax2.grid(True)

plt.tight_layout()
plt.savefig('plots/' + name + '.png', bbox_inches="tight")