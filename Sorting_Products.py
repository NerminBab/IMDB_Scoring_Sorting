import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5' % x)

df = pd.read_csv("", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]

df.head()