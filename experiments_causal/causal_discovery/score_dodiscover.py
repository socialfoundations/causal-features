"""Python script to run SCORE algorithm.

We use the python library "dodiscover", which is not officially released yet. (https://github.com/py-why/dodiscover)
"""
#%%
from pywhy_graphs.viz import draw
from dodiscover.ci import GSquareCITest
from dodiscover import make_context
from dodiscover.constraint import PC, FCI
import pickle
import pandas as pd
# import numpy as np
# import networkx as nx
# from scipy import stats
from dodiscover.toporder import CAM, SCORE, DAS, NoGAM


experiment = "acsunemployment"
experiment_name = "unemployment"

# assume additive gaussian noise model
data = pd.read_csv(f"/home/vnastl/causal-features/tmp_preprocessed/{experiment_name}.csv").astype("float32")
data = data.sample(n=1000, random_state=0)
data = data.loc[:,data.apply(pd.Series.nunique) != 1]

#%% dodiscover package
context = make_context().variables(data=data).build()

dd_score = SCORE()
dd_score.learn_graph(data, context)
with open(f"/home/vnastl/causal-features/tmp_preprocessed/dodiscover_score_{experiment_name}.pickle", 'wb') as handle:
    pickle.dump(dd_score, handle)

graph = dd_score.graph_

dot_graph = draw(graph)