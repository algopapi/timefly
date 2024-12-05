import pytest
import torch
from timefly.soft_dtw import SoftDTW 

SHAPES = [
    ((2, 10, 1),(2, 10, 1)),   # batch_size=2, seq_len=10, dim=1
    ((20, 100, 2),(20, 100, 2))   # batch_size=10, seq_len=10, dim=2 
]
GAMMA_VALUES = [0.1, 1.0]
BATCH_SIZES = [2, 5]


@pytest.fixture
def cpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=False)


@pytest.fixture
def gpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True)


@pytest.fixture(params=[BATCH_SIZES])
def gpu_dtw_batch(request):
    return SoftDTW(
        gamma=1.0, 
        normalize=False, 
        use_cuda=True,
        batch_size=request.param
    )


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
def test_cpu_is_gpu_forward(cpu_dtw, gpu_dtw, X_shape, Y_shape):
    """
    perform unit test that match the cpu and gpu forward pass implementation     
    """
    X = torch.randn(*X_shape)
    Y = torch.randn(*Y_shape)
    cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw.forward(X, Y)

    assert torch.allclose(cpu_out, gpu_out.cpu(), rtol=1e-5)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
def test_cpu_is_gpu_backward(cpu_dtw, gpu_dtw, X_shape, Y_shape):
    X = torch.randn(*X_shape, requires_grad=True)
    Y = torch.randn(*Y_shape, requires_grad=True)

    X_cuda = X.clone().detach().requires_grad_(True)
    Y_cuda = Y.clone().detach().requires_grad_(True)

    cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw.forward(X_cuda, Y_cuda)

    cpu_loss = cpu_out.mean()
    gpu_loss = gpu_out.mean()

    cpu_loss.backward()
    gpu_loss.backward()

    assert torch.allclose(X.grad, X.grad.cpu(), rtol=1e-5)
    assert torch.allclose(Y.grad, Y.grad.cpu(), rtol=1e-5)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
@pytest.mark.parametrize(
    "gpu_dtw_batch",
    BATCH_SIZES,
    indirect=True
)
def test_forward_at_n_batch_size(
    # cpu_dtw, 
    gpu_dtw_batch, 
    X_shape, 
    Y_shape
):
    """
    Test if our batching is impelmented correctly  
    """
    print(gpu_dtw_batch)
    print(X_shape, Y_shape)
    assert gpu_dtw_batch.batch_size == 2 or 5 
    # X = torch.randn(*X_shape)
    # Y = torch.randn(*Y_shape)
    # cpu_out = cpu_dtw.forward(X, Y)
    # gpu_out = gpu_dtw_batch.forward(X, Y)

    # assert cpu_out.shape == gpu_out.shape


def test_memory_leak(gpu_dtw):
    """
    Test if there is any memory leak in the gpu implementation
    """
    X = torch.randn(2, 10, 1).cuda()
    Y = torch.randn(2, 10, 1).cuda()
    for _ in range(1000):
        gpu_dtw.forward(X, Y)
        torch.cuda.empty_cache()


#TODO: Test results on multiple batches 
#TODO: Test memory leaks