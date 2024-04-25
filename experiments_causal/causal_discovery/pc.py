"""Python script to run pc algorithm."""
#%%
import pickle
from causallearn.search.ConstraintBased.PC import pc
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

# %% causallearn package
cg = pc(data, indep_test="gsq")
with open(f"/home/vnastl/causal-features/tmp_preprocessed/causallearn_pc_{experiment_name}.pickle", 'wb') as handle:
    pickle.dump(cg, handle)




