import pytest 
import torch
import numpy as np

from timefly.kmeans import TimeSeriesKMeans

SHAPES = [(2, 10, 1)]

import pytest
import torch
import numpy as np

from timefly.kmeans import TimeSeriesKMeans

@pytest.fixture
def synthetic_data():
    # Create synthetic time series data
    torch.manual_seed(0)
    np.random.seed(0)
    n_samples = 50
    seq_len = 20
    n_features = 1
    X = torch.rand(n_samples, seq_len, n_features, requires_grad=True).to(device='cuda')
    return X

def test_minibatch_vs_fullbatch_centroid_update(synthetic_data):
    """
    Test that minibatched centroid updates converge to roughly the same centroids
    as full batch centroid updates.
    """
    # Parameters
    n_clusters = 2  # Use 1 cluster for simplicity
    gamma = 1.0
    lr = 0.1
    max_iter_barycenter = 10
    num_iters = 5  # Number of iterations for centroid update

    # Initialize TimeSeriesKMeans instances
    seed = 0
    random_state = np.random.RandomState(seed)

    # Instance with full batch updates
    kmeans_fullbatch = TimeSeriesKMeans(
        n_clusters=n_clusters,
        gamma=gamma,
        device='cuda',
        random_state=random_state,
        optimizer='adam',
        optimizer_kwargs={'lr': lr},
    )
    kmeans_fullbatch.update_bs = synthetic_data.shape[0]  # Batch size equal to dataset size

    # Instance with minibatch updates
    kmeans_minibatch = TimeSeriesKMeans(
        n_clusters=n_clusters,
        gamma=gamma,
        device='cuda',
        random_state=random_state,
        optimizer='adam',
        optimizer_kwargs={'lr': lr},
    )
    kmeans_minibatch.update_bs = 10  # Set minibatch size

    # Use the same initial centroid
    initial_center = synthetic_data[0].clone().to(device='cuda')
    
    # Cluster assignment (all data assigned to cluster 0)
    labels = torch.zeros(synthetic_data.shape[0], dtype=torch.long, device='cuda')

    # Perform centroid updates
    X_cluster = synthetic_data[labels == 0]

    # Full batch centroid update
    centroid_fullbatch = kmeans_fullbatch._update_centroid(
        X_cluster, init_center=initial_center
    )

    # Minibatch centroid update
    centroid_minibatch = kmeans_minibatch._update_centroid(
        X_cluster, init_center=initial_center
    )

    # Compare the centroids
    difference = torch.norm(centroid_fullbatch - centroid_minibatch).item()
    print(f"Difference between full batch and minibatch centroids: {difference}")

    tolerance = 1e-3
    assert difference < tolerance, (
        f"Centroids diverged significantly between full batch and minibatch updates "
        f"(difference: {difference}, tolerance: {tolerance})."
    )