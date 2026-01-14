import os
from typing import Optional

import torch
from torch import distributed as dist

def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False

def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    if dist_backend is None:
        # FIXME sane defaults for other device backends?
        dist_backend = 'nccl' if 'cuda' in device else 'gloo'
    dist_url = dist_url or 'env://'

    # TBD, support horovod?
    # if args.horovod:
    #     import horovod.torch as hvd
    #     assert hvd is not None, "Horovod is not installed"
    #     hvd.init()
    #     args.local_rank = int(hvd.local_rank())
    #     args.rank = hvd.rank()
    #     args.world_size = hvd.size()
    #     args.distributed = True
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    #     os.environ['RANK'] = str(args.rank)
    #     os.environ['WORLD_SIZE'] = str(args.world_size)
    if is_distributed_env():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if 'cuda' in device:
        assert torch.cuda.is_available(), f'CUDA is not available but {device} was specified.'

    if distributed and device != 'cpu':
        device, *device_idx = device.split(':', maxsplit=1)

        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            _logger.warning(f'device index {device_idx[0]} removed from specified ({device}).')

        device = f'{device}:{local_rank}'

    if device.startswith('cuda:'):
        torch.cuda.set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )