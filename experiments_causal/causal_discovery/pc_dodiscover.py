"""Python script to run pc algorithm.

We use the python library "dodiscover", which is not officially released yet. (https://github.com/py-why/dodiscover)
"""
#%%
from pywhy_graphs.viz import draw
from dodiscover.ci import GSquareCITest
from dodiscover import make_context
from dodiscover.constraint import PC, FCI
import pickle
# import numpy as np
# import networkx as nx
# from scipy import stats

# from dodiscover.toporder import CAM, SCORE, DAS, NoGAM


import pandas as pd
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed

experiment = "bfrss_diabetes"
experiment_name = "diabetes"

data = pd.read_csv(f"/home/vnastl/causal-features/tmp_preprocessed/{experiment_name}_discrete_5.csv").astype("int")

#%% dodiscover package
context = make_context().variables(data=data).build()

ci_estimator = GSquareCITest(data_type="discrete")
dd_pc = PC(ci_estimator=ci_estimator)
dd_pc.learn_graph(data, context)
with open(f"/home/vnastl/causal-features/tmp_preprocessed/dodiscover_pc_{experiment_name}.pickle", 'wb') as handle:
    pickle.dump(dd_pc, handle)

graph = dd_pc.graph_

dot_graph = draw(graph)
dot_graph.render(outfile=f"/home/vnastl/causal-features/tmp_preprocessed/ci_cpdag.png")
