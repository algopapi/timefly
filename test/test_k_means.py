import pytest 
import torch
import numpy as np

from timefly.kmeans import TimeSeriesKMeans

SHAPES = [(2, 10, 1)]

@pytest.fixture()
def k_means():
    """
    Initiliaze the k means objet with a fixed zero seed 
    """
    seed = 0
    random_state = np.random.RandomState(seed)
    return TimeSeriesKMeans(
       n_clusters=3,
       max_iter=10,
       tol=1e-6,
       gamma=1,
       device='cuda',
       n_init=1,
       random_state=random_state, 
       optimizer='adam',
       optimizer_kwargs = {'lr': 0.1, 'max_iter': 20}
    ) 


@pytest.mark.parametrize("X_shape", SHAPES)
def test_k_means_init(k_means, X_shape):
    """
    K means init is a intialization sequence that guesses
    the initial centroids of the cluster.  
    """
    torch.manual_seed(0)
    X = torch.rand(*X_shape).to(device='cuda')

    print("X", X)
    k_means._k_means_init(X=X)


def test_fit_one_step():
    pass


def test_convergence():
    pass