import pytest
import torch
from timefly.soft_dtw import SoftDTW 

SHAPES = [
    ((2, 10, 1),(2, 10, 1)),   # batch_size=2, seq_len=10, dim=1
    ((20, 100, 1),(20, 100, 1))   # batch_size=10, seq_len=10, dim=2 
]

SHAPES_MEDIUM = [
    ((100, 100, 1),(100, 100, 1)),   # batch_size=2, seq_len=10, dim=1
]

GAMMA_VALUES = [0.1, 1.0]
BATCH_SIZES = [2, 5]


@pytest.fixture
def cpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=False, requires_grad=True)


@pytest.fixture
def gpu_dtw_no_grad():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True, requires_grad=False)


@pytest.fixture
def gpu_dtw_grad():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True, requires_grad=True)


@pytest.fixture(params=[BATCH_SIZES])
def gpu_dtw_batch(request):
    print("request.param", request.param)
    return SoftDTW(
        gamma=1.0, 
        normalize=False, 
        use_cuda=True,
        batch_size=request.param,
        requires_grad=True,
    )


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
def test_cpu_is_gpu_forward(cpu_dtw, gpu_dtw_no_grad, X_shape, Y_shape):
    """
    perform unit test that match the cpu and gpu forward pass implementation     
    """
    X = torch.randn(*X_shape)
    Y = torch.randn(*Y_shape)
    cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw_no_grad.forward(X, Y)

    assert torch.allclose(cpu_out, gpu_out.cpu(), rtol=1e-5)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
def test_cpu_is_gpu_backward(cpu_dtw, gpu_dtw_grad, X_shape, Y_shape):
    X = torch.randn(*X_shape, requires_grad=True)
    Y = torch.randn(*Y_shape, requires_grad=True)
    
    X_cuda = X.clone().detach().requires_grad_(True)
    Y_cuda = Y.clone().detach().requires_grad_(True)

    cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw_grad.forward(X_cuda, Y_cuda)

    cpu_loss = cpu_out.mean()
    gpu_loss = gpu_out.mean()

    cpu_loss.backward()
    gpu_loss.backward()
    
    assert torch.allclose(X.grad, X.grad.cpu(), rtol=1e-3)
    assert torch.allclose(Y.grad, Y_cuda.grad.cpu(), rtol=1e-3)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
@pytest.mark.parametrize(
    "gpu_dtw_batch",
    BATCH_SIZES,
    indirect=True
)
def test_forward_at_n_batch_size(
    cpu_dtw, 
    gpu_dtw_batch, 
    X_shape, 
    Y_shape
):
    """
    Test if our batching is impelmented correctly  
    """
    print(gpu_dtw_batch)
    X = torch.randn(*X_shape)
    Y = torch.randn(*Y_shape)
    cpu_out = cpu_dtw.forward(X, Y)
    gpu_out = gpu_dtw_batch.forward(X, Y)
    assert cpu_out.shape == gpu_out.shape
    assert torch.allclose(cpu_out, gpu_out.cpu(), rtol=1e-5)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES_MEDIUM)
def test_forward_memory_consistancy(gpu_dtw_no_grad, X_shape, Y_shape):
    """
    Test if there is any memory leak in the gpu implementation
    """
    allocation_before = torch.cuda.memory_allocated()

    X = torch.randn(*X_shape)
    Y = torch.randn(*Y_shape)

    forward = gpu_dtw_no_grad.forward(X, Y)
    allocation_after = torch.cuda.memory_allocated()
    assert abs(allocation_before - allocation_after) < 1000, "Forward mem leak"

    loss = forward.mean()
    loss.backward()

    allocation_after = torch.cuda.memory_allocated()
    assert abs(allocation_before - allocation_after) < 1000, "Backward mem leak"


def test_cdist_memory():

    torch.cuda.empty_cache()
    start_alloc = torch.cuda.memory_allocated()

    with torch.no_grad(): 
        X = torch.randn(1000, 100, 1, requires_grad=False).to(device='cuda')
        Y = torch.randn(1000, 100, 1, requires_grad=False).to(device='cuda')
        D_xy = torch.cdist(X, Y).pow(2)
        print(torch.cuda.memory_allocated())

    del D_xy, X, Y 
    torch.cuda.empty_cache()    
    print(torch.cuda.memory_allocated())
    
    assert torch.cuda.memory_allocated() <= start_alloc, "Memory leak"
    