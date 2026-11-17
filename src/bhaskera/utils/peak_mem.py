"""Peak GPU memory across all ranks, for the step log (sizes batches / checkpointing against the 80 GB)."""
import torch
import torch.distributed as dist


def peak_mem_gib(device) -> tuple[float, float]:
    """(max allocated, max reserved) in GiB, the maximum over all ranks since start. Collective: all ranks call."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    t = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                     dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return tuple((t / 2**30).tolist())
