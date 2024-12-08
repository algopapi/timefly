# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ---------------------------------------------------------------------------------------------------------------------- #
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


# ------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """
    @staticmethod
    def forward(ctx, D, gamma, bandwidth, requires_grad):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        softdtw = compute_softdtw(D_, g_, b_)
        R = torch.Tensor(softdtw).to(dev).type(dtype)
        if requires_grad:
            ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None


class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """
    @staticmethod
    def forward(ctx, D, gamma, bandwidth, requires_grad):   
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D.detach()),
            gamma.item(), 
            bandwidth.item(), 
            N, 
            M, 
            n_passes,
            cuda.as_cuda_array(R)
        )

        # only save for backward if requires grad is True
        if requires_grad:
            ctx.save_for_backward(D, R.clone(), gamma, bandwidth)

        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1
 
        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R),
            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
            cuda.as_cuda_array(E)
        )
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """
    def __init__(
            self, 
            use_cuda:bool=True, 
            gamma:float=1.0, 
            batch_size:float=3e3,
            normalize:bool=False, 
            bandwidth:float=None, 
            requires_grad:bool=False,
            device=None
        ):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        self.requires_grad = requires_grad
        self.requires_grad_(requires_grad)
        self.device = device if device is not None else torch.device('cuda' if use_cuda else 'cpu')

        # set distance function
        self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        return torch.cdist(x, y, p=2).pow(2)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        # check if it is a tensor
        if not isinstance(X, torch.Tensor): 
            X = torch.from_numpy(X)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y)

        # TODO: Add precision support
        if self.use_cuda:
            X = X.to(self.device)
            Y = Y.to(self.device) 

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        output = torch.empty(X.shape[0], dtype=X.dtype, device=self.device)
        with torch.set_grad_enabled(self.requires_grad):
           for i in range(0, X.shape[0], self.batch_size):
                end_i = min(X.shape[0], i + self.batch_size)
                D_i = self.dist_func(X[i:end_i], Y[i:end_i]) #(batch_size, seq_len_1, seq_len_2) 
                output[i:end_i]= func_dtw(D_i, self.gamma, self.bandwidth, True)

        # del D_i, X, Y
        # gc.collect()
        # torch.cuda.empty_cache()
        return output 


class PairwiseSoftDTW(torch.nn.Module):
    def __init__(
            self, 
            gamma: float = 1.0, 
            precision=torch.float32,
            use_cuda: bool = True
        ):
        super(PairwiseSoftDTW, self).__init__()
        self.gamma = gamma
        self.precision = precision
        self.bandwidth = 0
        self.use_cuda = cuda

    @staticmethod
    def _batch_euclidean_dist(A, B):
        """
        Efficiently compute pairwise Euclidean distance matrices between all sequences in A and B.

        Args:
            A: Tensor of shape (n_a, seq_len, dims)
            B: Tensor of shape (n_b, seq_len, dims)

        Returns:
            D: Tensor of shape (n_a, n_b, seq_len, seq_len)
        """
        # Expand A and B for broadcasting
        A_exp = A.unsqueeze(1).unsqueeze(3)   # Shape: (n_a, 1, seq_len, 1, dims)
        B_exp = B.unsqueeze(0).unsqueeze(2)   # Shape: (1, n_b, 1, seq_len, dims)
    
        # Compute pairwise squared differences
        diff = A_exp - B_exp  # Shape: (n_a, n_b, seq_len, seq_len, dims)
        dist_sq = (diff ** 2).sum(-1)  # Sum over feature dimensions -> Shape: (n_a, n_b, seq_len, seq_len)

        return dist_sq  # Shape: (n_a, n_b, seq_len, seq_len)


    def forward(self, X, Y):
        """
        input args:
            X: Source timeseries of shape  (n series, length, feature dim) 
            Y: Target time series of shape (n clusters, length, feauture dim)
        output:
            Dist: (n_series x n_cluster, dim) tensor of distances between target series and cluster series 
        """
        # check if it is  actually a tensor
        if not isinstance(X, torch.Tensor): 
            X = torch.from_numpy(X).to(dtype=self.precision, device='cuda')
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).to(dtype=self.precision, device='cuda')

        n_a, t_x, d_x = X.shape
        n_b, t_y, d_y = Y.shape

        assert t_x == t_y
        assert d_x == d_y
        t = t_x 
        
        D = self._batch_euclidean_dist(A=X, B=Y)
        D_flat = D.view(-1, t, t)

        func_dtw = _SoftDTWCUDA.apply
        distances = func_dtw(
            D_flat, 
            self.gamma, 
            self.bandwidth, 
            False
        )

        D_dist = distances.view(n_a, n_b) 
        return D_dist 


    def forward_batched(self, X, Y, chunk_size = 1000):
        """
        Args:
            X: Source timeseries of shape (n_a, length, feature_dim) 
            Y: Target time series of shape (n_b, length, feature_dim)
        Returns:
            D_dist: (n_a, n_b) tensor of distances between each X_i and Y_j
        """
        # Ensure inputs are tensors on the correct device
        device = 'cuda' if self.use_cuda else 'cpu'
        if not isinstance(X, torch.Tensor): 
            X = torch.from_numpy(X).to(dtype=self.precision, device=device)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).to(dtype=self.precision, device=device)

        n_a, t_x, d_x = X.shape
        n_b, t_y, d_y = Y.shape
        assert t_x == t_y, "Time lengths of X and Y must match"
        assert d_x == d_y, "Feature dimensions of X and Y must match"

        # We'll accumulate results here
        D_dist = torch.empty((n_a, n_b), dtype=self.precision, device=device)

        # Process Y in chunks to reduce memory footprint
        for start_idx in range(0, n_b, chunk_size):
            end_idx = min(start_idx + chunk_size, n_b)
            Y_chunk = Y[start_idx:end_idx]  # shape: (chunk_size, t, d)

            # Compute pairwise distances for this chunk
            D = self._batch_euclidean_dist(X, Y_chunk)  # (n_a, chunk_size, t, t)

            # Flatten for softDTW computation
            t = t_x
            D_flat = D.view(-1, t, t)

            func_dtw = _SoftDTWCUDA.apply
            distances = func_dtw(D_flat, self.gamma, self.bandwidth, False)
            # Reshape the result
            D_chunk_dist = distances.view(n_a, -1)  # (n_a, chunk_size)

            # Store in the result tensor
            D_dist[:, start_idx:end_idx] = D_chunk_dist

        return D_dist


class BatchedSoftDTW(torch.nn.Module):
    def __init__(
        self, 
        gamma=1.0, 
        precision=torch.float32, 
        use_cuda=True, 
        chunk_size=100
    ):
        super().__init__()
        self.gamma = gamma
        self.precision = precision
        self.bandwidth = 0
        self.use_cuda = use_cuda
        self.chunk_size = chunk_size

    @staticmethod
    def _elementwise_euclidean_dist(X, Y):
        """
        Compute elementwise Euclidean distance matrices between corresponding pairs X[i], Y[i].
        X, Y: (N, t, d)
        Returns: D (N, t, t)
        """
        return torch.cdist(X, Y, p=2).pow(2)

    @staticmethod
    def _pairwise_euclidean_dist(A, B):
        """
        Compute pairwise Euclidean distance matrices for all pairs (A[i], B[j]).
        A: (n_a, t, d)
        B: (n_b, t, d)
        Returns: (n_a, n_b, t, t)
        """
        A_exp = A.unsqueeze(1).unsqueeze(3)  # (n_a, 1, t, 1, d)
        B_exp = B.unsqueeze(0).unsqueeze(2)  # (1, n_b, 1, t, d)
        diff = A_exp - B_exp                  # (n_a, n_b, t, t, d)
        dist_sq = (diff ** 2).sum(-1)         # (n_a, n_b, t, t)
        return dist_sq

    def _soft_dtw(self, D_flat, with_grads):
        """
        Apply Soft-DTW on a batch of cost matrices.
        D_flat: (N, t, t)
        Returns: distances (N,)
        """
        # Decide which kernel to use based on device
        if self.use_cuda:
            return _SoftDTWCUDA.apply(D_flat, self.gamma, self.bandwidth, with_grads)
        else:
            return _SoftDTW.apply(D_flat, self.gamma, self.bandwidth, with_grads)

    def elementwise(self, X, Y, with_grads=True):
        """
        Computes Soft-DTW for each pair (X[i], Y[i]) -> returns (N,) distances.
        X, Y: (N, t, d)

        If with_grads=True, gradients are tracked. Otherwise no gradients are stored.
        """
        device = 'cuda' if self.use_cuda else 'cpu'
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(device=device, dtype=self.precision)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).to(device=device, dtype=self.precision)

        n_x, t_x, d_x = X.shape
        n_y, t_y, d_y = Y.shape
        assert t_x == t_y, "Time lengths must match."
        assert d_x == d_y, "Feature dimensions must match."
        assert n_x == n_y, "Elementwise mode requires X and Y to have the same batch size."

        grad_context = torch.enable_grad() if with_grads else torch.no_grad()
        with grad_context:
            distances_all = []
            for start_idx in tqdm(range(0, n_x, self.chunk_size), desc="Processing elementwise chunks", unit="chunk"):
                end_idx = min(start_idx + self.chunk_size, n_x)
                X_chunk = X[start_idx:end_idx]
                Y_chunk = Y[start_idx:end_idx]

                D = self._elementwise_euclidean_dist(X_chunk, Y_chunk)  # (chunk_size, t, t)
                distances_chunk = self._soft_dtw(D, with_grads)          # (chunk_size,)
                distances_all.append(distances_chunk)

            distances = torch.cat(distances_all, dim=0)  # (N,)
            return distances

    def pairwise(self, X, Y, with_grads=False):
        """
        Computes Soft-DTW for all pairs (X[i], Y[j]) -> returns (n_a, n_b).
        X: (n_a, t, d)
        Y: (n_b, t, d)

        If with_grads=False, no gradients are tracked for these computations.
        """
        device = 'cuda' if self.use_cuda else 'cpu'
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(device=device, dtype=self.precision)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).to(device=device, dtype=self.precision)

        n_a, t_x, d_x = X.shape
        n_b, t_y, d_y = Y.shape
        assert t_x == t_y, "Time lengths must match."
        assert d_x == d_y, "Feature dimensions must match."

        grad_context = torch.enable_grad() if with_grads else torch.no_grad()
        with grad_context:
            D_dist = torch.empty((n_a, n_b), dtype=self.precision, device=device)

            # process Y in chunks to reduce memory usage
            for start_idx in tqdm(range(0, n_b, self.chunk_size), desc="Processing pairwise distances", unit="chunk"):
                end_idx = min(start_idx + self.chunk_size, n_b)
                Y_chunk = Y[start_idx:end_idx]
                D = self._pairwise_euclidean_dist(X, Y_chunk)  # (n_a, chunk_size, t, t)
                D_flat = D.view(-1, t_x, t_x)
                dist_chunk = self._soft_dtw(D_flat, with_grads) # (n_a * chunk_size)
                D_chunk_dist = dist_chunk.view(n_a, -1)         # (n_a, chunk_size)
                D_dist[:, start_idx:end_idx] = D_chunk_dist
            return D_dist 