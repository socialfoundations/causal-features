"""Python script to get distribution shift metric and plot correlation of distribution shift and causal feature performance."""

from pathlib import Path
from tableshift import get_dataset
import torch
import torch.utils.data as data_utils
from otdd.pytorch.distance import DatasetDistance

import warnings
warnings.filterwarnings("ignore")

def main(
        experiment: str,
        cache_dir: str):
    """Get metric of covariate shift and label shift.

    Parameters
    ----------
    experiment : str
        The name of the experiment to run.
    cache_dir : str
        Directory to cache raw data files to.

    Returns
    -------
    dictionary.

    """
    cache_dir = Path(cache_dir)

    dset = get_dataset(experiment, cache_dir)
    features_id, target_id, _, _ = dset.get_pandas("validation")
    features_ood, target_ood, _, _ = dset.get_pandas("ood_validation")

    # Get values of Dataframe and Series
    target_id = target_id.values
    features_id = features_id.values
    target_ood = target_ood.values
    features_ood = features_ood.values

    # Convert the features and target to PyTorch tensors
    features_id_tensor = torch.tensor(features_id, dtype=torch.float32)
    target_id_tensor = torch.tensor(target_id, dtype=torch.int)
    features_ood_tensor = torch.tensor(features_ood, dtype=torch.float32)
    target_ood_tensor = torch.tensor(target_ood, dtype=torch.int)

    # Create a TensorDataset
    dataset_id = data_utils.TensorDataset(features_id_tensor, target_id_tensor)
    dataset_ood = data_utils.TensorDataset(features_ood_tensor, target_ood_tensor)

    # Save distances in dictionary
    d = dict()
    # Get Optimal transport datashift distance
    dist = DatasetDistance(dataset_id, dataset_ood)
    d["ot_datashift_distance"] = dist.distance()

    d["label_l2_distance"] = torch.cdist(target_id_tensor,target_ood_tensor,p=2.0)

    return d


    
if __name__ == "__main__":
    d = main(experiment="acsincome", cache_dir="tmp")
    print(d)