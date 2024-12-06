import torch
import gc

torch.cuda.empty_cache()
start_alloc = torch.cuda.memory_allocated()
print("start alloc", start_alloc)

for i in range (10):
    X = torch.randn(1000, 100, 1, requires_grad=False).to(device='cuda')
    Y = torch.randn(1000, 100, 1, requires_grad=False).to(device='cuda')
    with torch.no_grad(): 
        D_xy = torch.cdist(X, Y).pow(2)

    # print(torch.cuda.memory_allocated())
    # del D_xy, X, Y 
    # gc.collect()
    # torch.cuda.empty_cache()    
    print(torch.cuda.memory_allocated())

    # assert torch.cuda.memory_allocated() <= start_alloc, "Memory leak"
