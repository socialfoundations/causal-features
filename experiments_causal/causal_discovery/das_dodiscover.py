"""Python script to run pc algorithm.

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


experiment = "bfrss_diabetes"
experiment_name = "diabetes"

data = pd.read_csv(f"/home/vnastl/causal-features/tmp_preprocessed/{experiment_name}_discrete_5.csv").astype("int")

#%% dodiscover package
context = make_context().variables(data=data).build()

dd_das = DAS()
dd_das.learn_graph(data, context)
with open(f"/home/vnastl/causal-features/tmp_preprocessed/dodiscover_das_{experiment_name}.pickle", 'wb') as handle:
    pickle.dump(dd_das, handle)

graph = dd_das.graph_

dot_graph = draw(graph)
dot_graph.render(outfile=f"/home/vnastl/causal-features/tmp_preprocessed/das_{experiment_name}.png")
