"""Run with torchrun --standalone --nproc-per-node=2 -m exp.test_distributed."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from exp.optimizer import ExperimentOptimizer


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    torch.manual_seed(42)
    model = DistributedDataParallel(torch.nn.Linear(4, 3, bias=False))
    parameter = next(model.parameters())
    optimizer = ExperimentOptimizer([{"params": [parameter], "kind": "matrix"}])
    for step in range(4):
        torch.manual_seed(step * 10 + dist.get_rank())
        model(torch.randn(2, 4)).square().mean().backward()
        optimizer.observe_unclipped_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        snapshot = torch.cat([parameter.detach().flatten(),
                              optimizer.state[parameter]["meta"]["factor"].reshape(1),
                              optimizer.state[parameter]["meta"]["sum_sq"].reshape(1)])
        gathered = [torch.empty_like(snapshot) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, snapshot)
        for other in gathered:
            torch.testing.assert_close(snapshot, other, atol=0, rtol=0)
    if dist.get_rank() == 0:
        print("DDP parameters, hypergradient factors and AdaGrad states match exactly")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
