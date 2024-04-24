"""Python script to save experiments into csv files in different encodings.

We use the python library "dodiscover", which is not officially released yet. (https://github.com/py-why/dodiscover)
"""
#%%
import numpy as np
import networkx as nx
from scipy import stats
from pywhy_graphs.viz import draw
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover import make_context
from dodiscover.constraint import PC, FCI
from dodiscover.toporder import CAM, SCORE, DAS, NoGAM

from causallearn.search.ConstraintBased.PC import pc

import pandas as pd
from dowhy import gcm
from dowhy.gcm.util.general import set_random_seed

experiment = "bfrss_diabetes"
experiment_name = "diabetes"

data = pd.read_csv(f"/Users/vnastl/Seafile/My Library/mpi project causal vs noncausal/causal-features/tmp_preprocessed/{experiment_name}_discrete_5.csv").astype("int")

#%% dodiscover package
context = make_context().variables(data=data).build()

ci_estimator = GSquareCITest(data_type="discrete")
dd_pc = PC(ci_estimator=ci_estimator)
dd_pc.learn_graph(data, context)

graph = dd_pc.graph_

dot_graph = draw(graph)
dot_graph.render(outfile="ci_cpdag.png", view=True)

# %% causallearn package
cg = pc(data, indep_test="gsq")
