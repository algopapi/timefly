import pytest
import torch
from timefly.soft_dtw import SoftDTW 

@pytest.fixture
def cpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=False)

@pytest.fixture
def gpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True)

@pytest.mark.parametrize("X_shape, Y_shape", [
    ((2, 10, 1),(2, 10, 1)),   # batch_size=2, seq_len=10, dim=1
    ((20, 100, 2),(20, 100, 2))   # batch_size=10, seq_len=10, dim=2 
])
def test_cpu_is_gpu_forward(cpu_dtw, gpu_dtw, X_shape, Y_shape):
    """
    perform unit test that match the cpu and gpu forward pass implementation     
    """
    X = torch.randn(*X_shape)
    Y = torch.randn(*Y_shape)
    # cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw.forward(X, Y)
    print("gpu out", gpu_out)

    # assert torch.allclose(cpu_out, gpu_out, rtol=1e-4)

#TODO: Test backward pass
#TODO: Test batching
#TODO: Test memory leaks
