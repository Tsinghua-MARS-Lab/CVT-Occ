import itertools
import copy

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler
# from .nuscene_dataset_detail import new_train_data_scene_size_list, new_group_idx_to_sample_idxs
from .waymo_dataset_detail import train_data_group_flag, new_group_idx_to_sample_idxs

def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

class MyGroupBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.

    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 total_epochs=8,
                 load_interval=1,
                 train_process=True,):

        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.total_epochs = total_epochs

        self.final_data_group_size =  train_data_group_flag

        self.dataset = dataset
        self.batch_size = batch_size
        assert self.batch_size == 1, "warning! batch_size > 1"

        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        
        assert load_interval == 1, "not support load interval not equal to 1"
        self.load_interval = load_interval
        self.size = len(self.dataset)
        
        self.group_sizes = np.array(self.final_data_group_size)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        self.group_idx_to_sample_idxs = new_group_idx_to_sample_idxs

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [ self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)] # [[]]

    def _infinite_group_indices(self):
        total_group_indices_list = []
        g = torch.Generator()
        g.manual_seed(self.seed)
        for i in range(self.total_epochs + 1): # Here +1 for the last epoch # donot worry we only use the sampler to train. 
            each_epoch_group_indices_list = torch.randperm(self.groups_num, generator=g).tolist()
            total_group_indices_list += each_epoch_group_indices_list[:self.groups_num // self.load_interval] # self.groups_num // self.load_interval
        yield from total_group_indices_list

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    try:
                        new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    except StopIteration:
                        break
                    self.buffer_per_local_sample[local_sample_idx] = copy.deepcopy(self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            if curr_batch == []:
                return
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch
