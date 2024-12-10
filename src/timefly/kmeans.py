import torch
import numpy as np
from tqdm.auto import tqdm
from timefly.soft_dtw import BatchedSoftDTW


class TimeSeriesKMeans:
    """TimeSeries K-Means clustering using SoftDTW and PyTorch.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.

    max_iter : int, default=50
        Maximum number of iterations of the k-means algorithm.

    tol : float, default=1e-6
        Relative tolerance with regards to inertia to declare convergence.

    gamma : float, default=1.0
        SoftDTW gamma parameter.

    device : str, default='cuda'
        Device to use for computations ('cuda' or 'cpu').

    n_init : int, default=1
        Number of time the k-means algorithm will be run with different centroid seeds.

    Attributes
    ----------
    cluster_centers_ : torch.Tensor
        Cluster centers (barycenters), of shape (n_clusters, seq_len, n_features).

    labels_ : numpy.ndarray
        Labels of each time series.
    """
    def __init__(
            self, 
            n_clusters=3, 
            max_iter=50, 
            tol=1e-6, 
            gamma=1.0, 
            device='cuda', 
            n_init=1,
            random_state=None,
            optimizer = 'adam',
            optimizer_kwargs={'lr': 1.0}
        ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.gamma = gamma
        self.device = device
        self.n_init = n_init
        self.max_iter_barycenter = 10

        self.dtw = BatchedSoftDTW(
            gamma=self.gamma,
            chunk_size_pairwise=1000,
            chunk_size_elementwise=2000,
            use_cuda=True
        )
        self.optimizer = optimizer 
        self.optimizer_kwargs = optimizer_kwargs 
        self.update_bs = 1000

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert data to torch tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        best_inertia = float('inf')
        best_labels = None
        best_centers = None

        for init_no in range(self.n_init):
            labels, inertia, centers = self._fit_one_init(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        self.labels_ = best_labels.cpu().numpy()
        self.cluster_centers_ = best_centers.detach()
        return self

    def predict(self, X):
        """
        Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len, n_features)
            New data to predict.

        Returns
        -------
        labels : numpy.ndarray
            Index of the cluster each sample belongs to.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # distances = self.distance(X, self.cluster_centers_)
        distances = self.dtw.pairwise(self.cluster_centers_, X, with_grads=False).transpose(1, 0)

        labels = torch.argmin(distances, dim=1)

        return labels.cpu().numpy()

    def _fit_one_init(self, X):
        """
        Initialize cluster centers and perform clustering with one initialization.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        random_state : int or None
            Seed for random number generator.

        Returns
        -------
        labels : torch.Tensor
            Labels of each time series.
        inertia : float
            Sum of distances (inertia) for assigned clusters.
        centers : torch.Tensor
            Cluster centers.
        """
        n_samples = X.shape[0]

        # Initialize cluster centers using k-means++
        cluster_centers = self._k_means_init(X).clone().requires_grad_(True)

        for i in tqdm(
            range(self.max_iter), 
            desc="K-means iteration", 
            unit="iter", 
            leave=True, 
            position=0
        ):
            # Compute distances between each time series and each cluster center
            # distances = self.distance(X, cluster_centers)
            distances = self.dtw.pairwise(cluster_centers, X, with_grads=False).transpose(1, 0)
            # assert torch.allclose(distances, distances_2)

            # Assign each time series to the nearest cluster center
            labels = torch.argmin(distances, dim=1)

            # Compute inertia (sum of distances for assigned clusters)
            inertia = torch.sum(
                distances[torch.arange(n_samples), labels].pow(2)
            ) / n_samples

            # Update cluster centers
            new_centers = []
            for k in tqdm(
                range(self.n_clusters),
                desc="Updating centroids",
                unit="cluster",
                leave=True,
                position=1
            ):
                X_k = X[labels == k] # shape  = (t, d)
                init_center = cluster_centers[k]
                if X_k.nelement() == 0:
                    new_center = X[torch.randint(0, n_samples, (1,))].squeeze(0)
                else:
                    new_center = self._update_centroid(X_k, init_center=init_center)

                new_centers.append(new_center.detach())

            new_centers = torch.stack(new_centers)
            center_shift = torch.norm(cluster_centers - new_centers)
            
            # Convergence?
            if center_shift < self.tol:
                print("Detected minimal center shifts... Converged...")
                break

            cluster_centers = new_centers.clone().detach().requires_grad_(True)

        return labels, inertia.item(), cluster_centers

    def _k_means_init(self, X):
        """
        Initialize cluster centers using the k-means++ algorithm.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, n_features)
            Training data.

        random_state : numpy.RandomState
            Random number generator.

        Returns
        -------
        centers : torch.Tensor of shape (n_clusters, seq_len, n_features)
            Initialized cluster centers.
        """
        n, t, d = X.shape
        n_clusters = self.n_clusters

        n_local_trials = 2 + int(np.log(n_clusters))
        centers = torch.empty(
            (n_clusters, t, d), 
            dtype=X.dtype, 
            device=X.device
        )

        # Choose the first center using NumPy's random_state
        c_id = self.random_state.randint(0, n)
        centers[0] = X[c_id]

        # Initialize list of squared distances to closest center
        # closest_dist_sq = self.distance(centers[0].unsqueeze(0), X) ** 2
        closest_dist_sq = self.dtw.pairwise(centers[0].unsqueeze(0), X, with_grads=False).pow(2)

        # assert torch.allclose(closest_dist_sq, closest_dist_sq_2)

        current_pot = closest_dist_sq.sum().item()

        for c in tqdm(
            range(1, n_clusters),
            desc="Initializing centers...",
            unit="cluster",
            leave=True,
            position=0
        ):
            # Generate rand_vals using NumPy's random_state
            rand_vals_np = self.random_state.random_sample(n_local_trials) * current_pot

            # Convert rand_vals to PyTorch tensor on GPU with appropriate dtype
            rand_vals = torch.from_numpy(rand_vals_np).to(
                device=X.device, 
                dtype=closest_dist_sq.dtype
            )

            # Compute cumulative sum of distances
            c_ids = torch.searchsorted(
                torch.cumsum(closest_dist_sq.flatten(), dim=0), 
                rand_vals
            )

            max = closest_dist_sq.size(1) -1
            c_ids = torch.clamp(c_ids, min=None, max=max)

            # Compute distances to center candidates
            # distance_to_candidates = self.distance(X[c_ids], X) ** 2
            distance_to_candidates = self.dtw.pairwise(X[c_ids], X, with_grads=False).pow(2)
            # assert torch.allclose(distance_to_candidates, distance_to_candidates_2)

            # Update closest distances squared and potential for each candidate
            closest_dist_sq_candidate = torch.minimum(
                closest_dist_sq, 
                distance_to_candidates
            )

            candidates_pot = closest_dist_sq_candidate.sum(dim=1)

            # Decide which candidate is the best
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate].item()
            closest_dist_sq = closest_dist_sq_candidate[best_candidate].unsqueeze(0)
            best_candidate_id = c_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers[c] = X[best_candidate_id]

        return centers 

    def _update_centroid(self, X_cluster, init_center):
        num_iters = 10
        lr = 0.1
        centroid = init_center.clone().detach().requires_grad_(True)

        if self.optimizer.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [centroid], 
                lr=self.optimizer_kwargs.get('lr', 1.0), 
                max_iter=self.max_iter_barycenter,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                centroid_expanded = centroid.unsqueeze(0).expand(X_cluster.shape[0], -1, -1)
                #sdtw_values = self.barycenter(centroid_expanded, X_cluster)
                sdtw_values = self.dtw.elementwise(centroid_expanded, X_cluster, with_grads=True)
                loss = sdtw_values.mean()
                loss.backward()
                return loss
            optimizer.step(closure)

        else: 
            optimizer = torch.optim.Adam([centroid], lr=lr)
            for _ in tqdm(
                range(num_iters), 
                desc="centroid update step", 
                unit="step", 
                leave=True, 
                position=2
            ):
                optimizer.zero_grad()
                total_loss = 0.0
                n_samples = X_cluster.shape[0]                
                for i in tqdm(
                    range(0, n_samples, self.update_bs), 
                    desc="batch update", 
                    unit="batch",
                    leave=True, 
                    position=3
                ):
                    end = min(i + self.update_bs, n_samples)
                    batch_X = X_cluster[i:end]
                    centroid_expanded = centroid.unsqueeze(0).expand(
                        batch_X.shape[0], -1, -1
                    )
                    sdtw_values = self.dtw.elementwise(centroid_expanded, batch_X, with_grads=True)
                    loss = sdtw_values.mean()
                    loss.backward()
                    total_loss += loss.item() * batch_X.shape[0]

                optimizer.step()

        return centroid.data