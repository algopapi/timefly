import pytest
import torch
from timefly.soft_dtw import SoftDTW, PairwiseSoftDTW, BatchedSoftDTW

GAMMA_VALUES = [0.01, 0.1, 1.0]
EFFECTIVE_BATCH_SIZES = [10, 100]

SHAPES = [
    ((2, 10, 1),(2, 10, 1)),   # batch_size=2, seq_len=10, dim=1
    ((20, 100, 1),(20, 100, 1))   # batch_size=10, seq_len=10, dim=2 
]

SHAPES_MEDIUM = [
    ((100, 100, 1),(100, 100, 1)),   # batch_size=2, seq_len=10, dim=1
]


@pytest.fixture
def cpu_dtw():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=False, requires_grad=True)


@pytest.fixture
def gpu_dtw_no_grad():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True, requires_grad=False)


@pytest.fixture
def gpu_dtw_grad():
    return SoftDTW(gamma=1.0, normalize=False, use_cuda=True, requires_grad=True)


@pytest.fixture
def dtw_pairwise():
    return PairwiseSoftDTW(
        gamma=1.0, 
        use_cuda=True, 
    ) 

@pytest.fixture
def dtw_batched():
    return BatchedSoftDTW(
        gamma=1.0, 
        use_cuda=True, 
        chunk_size=1000,
    )

@pytest.fixture(params=EFFECTIVE_BATCH_SIZES)
def gpu_dtw_batch(request):
    return SoftDTW(
        gamma=1.0, 
        normalize=False, 
        use_cuda=True,
        batch_size=request.param,
        requires_grad=False,
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

    assert torch.allclose(cpu_out, gpu_out.cpu(), rtol=1e-3)


def test_gpu_backward(gpu_dtw_grad):
    X = torch.randn(2, 10, 1, requires_grad=True)
    Y = torch.randn(2, 10, 1, requires_grad=True)
    gpu_out = gpu_dtw_grad.forward(X, Y)
    loss = gpu_out.mean()
    loss.backward()
    assert X.grad is not None
    assert Y.grad is not None


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

    cpu_loss.backward() # does invoke backward autograd
    gpu_loss.backward() # does not invoke backward autograd? at least not in debug mode
    
    assert torch.allclose(X.grad, X.grad.cpu(), rtol=1e-3)
    assert torch.allclose(Y.grad, Y_cuda.grad.cpu(), rtol=1e-3)


@pytest.mark.parametrize("X_shape, Y_shape", SHAPES)
@pytest.mark.parametrize(
    "gpu_dtw_batch",
    EFFECTIVE_BATCH_SIZES,
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


def test_regular_is_batched(gpu_dtw_grad, dtw_batched):
    X = torch.randn(2, 10, 1, device="cuda")
    Y = torch.randn(2, 10, 1, device="cuda")

    regular = gpu_dtw_grad.forward(X, Y)
    batched = dtw_batched.forward(X, Y, pairwise=False)

    assert torch.allclose(regular, batched, rtol=1e-3)


def test_pairwise_is_batched(dtw_pairwise, dtw_batched):
    X = torch.randn(30, 10, 1, device="cuda")
    Y = torch.randn(400, 10, 1, device="cuda")
    pairwise = dtw_pairwise.forward(X, Y)
    batched = dtw_batched.pairwise(X, Y, with_grads=False)
    assert torch.allclose(pairwise, batched, rtol=1e-4)


def test_batched_pairwise_large(dtw_batched):
    X = torch.randn(10, 1000, 1, device="cuda")# 10 clusters
    Y = torch.randn(40000, 1000, 1, device="cuda") #40K portoflios

    batched = dtw_batched.pairwise(X, Y, with_grads=False)
    assert batched.shape == (10, 40000)


def test_memory_leak_over_iterations(gpu_dtw_no_grad):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocation_before = torch.cuda.memory_allocated()

    for _ in range(2):
        X = torch.randn(1000, 1000, 1, device='cuda', requires_grad=False)
        Y = torch.randn(1000, 1000, 1, device='cuda', requires_grad=False)
        forward = gpu_dtw_no_grad.forward(X, Y)

        # Clean up
        del forward, X, Y
        torch.cuda.empty_cache()

    allocation_after = torch.cuda.memory_allocated()

    memory_growth = allocation_after - allocation_before
    print(f"Memory growth over iterations: {memory_growth} bytes")
    def __init__(self):
        self._start = torch.cuda.memory_allocated()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = torch.cuda.memory_allocated()
        self._delta = self._end - self._start
        print(f"Memory delta: {self._delta} bytes")
    acceptable_growth = 50 * 1024 * 1024  # 50 MB
    assert memory_growth < acceptable_growth, "Memory leak detected over iterations" 