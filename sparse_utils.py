from scipy import sparse
from torch import nn
import torch
from torch.nn.attention.flex_attention import BlockMask

import utils


class BlockSparseTensor:
    def __init__(self, physical: torch.Tensor, block_mask: BlockMask):
        self.block_mask = block_mask
        self.physical = physical
        self.block_size = block_mask.BLOCK_SIZE[0]
        self.kv_num_blocks = block_mask.kv_num_blocks
        self.kv_indices = block_mask.kv_indices

        self.lookup_table = self._build_lookup_table()
        ...

    @staticmethod
    def block_mask_to_index(block_mask: BlockMask):
        batch_size, _, n_blocks = block_mask.kv_num_blocks.shape
        block_size = block_mask.BLOCK_SIZE[0]
        n_tokens = block_size * n_blocks

        kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
        kv_inds_no_heads = block_mask.kv_indices[:, 0, :, :]
        num_blocks_per_batch = torch.sum(kv_num_blocks_no_heads, dim=-1)
        n_blocks_total = torch.sum(num_blocks_per_batch)

        # batch_num_blocks = [3, 4, 2] => batch_idx = [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
        batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_blocks_per_batch)
        batch_start_indices = torch.nn.functional.pad(
            torch.cumsum(num_blocks_per_batch, dim=0),
            (1, 0)
        )
        batch_starting_points = batch_start_indices[batch_idx]

        q_block_idx_total = torch.repeat_interleave(torch.arange(n_blocks), kv_num_blocks_no_heads.flatten())
        q_block_idx = q_block_idx_total - q_block_idx_total[batch_starting_points]
        q_start_indices = torch.nn.functional.pad(
            torch.cumsum(kv_num_blocks_no_heads.flatten(), dim=0),
            (1, 0)
        )
        q_block_starting_points = q_start_indices[q_block_idx_total]

        repeating_range = torch.arange(n_blocks_total)
        repeating_range = repeating_range - repeating_range[q_block_starting_points]

        k_block_idx = kv_inds_no_heads[batch_idx, q_block_idx, repeating_range]

        out_shape = (n_blocks_total, block_size, block_size)
        batch_idx = batch_idx.reshape(-1, 1, 1).broadcast_to(out_shape)
        q_block_idx = q_block_idx.reshape(-1, 1, 1).broadcast_to(out_shape)
        k_block_idx = k_block_idx.reshape(-1, 1, 1).broadcast_to(out_shape)

        q_within_block = torch.arange(block_size).reshape(1, -1, 1).broadcast_to(out_shape)
        k_within_block = torch.arange(block_size).reshape(1, 1, -1).broadcast_to(out_shape)

        q_full_idx = q_block_idx * block_size + q_within_block
        k_full_idx = k_block_idx * block_size + k_within_block
    
        return batch_idx, q_full_idx, k_full_idx

    @staticmethod
    def from_broadcast(x: torch.Tensor, block_mask: BlockMask, batch_shape):
        x = utils.unify_batch_dimension(x, batch_shape)

        if x.dim() == 3:
            # Add explicit feature dimension
            x = x.unsqueeze(-1)
        
        if x.dim() != 4:
            raise ValueError('BlockSparseTensors can only be constructed from tensors with dimension 2 or 3, excluding batch dimensions.')

        out_shape = (x.shape[0], )
        batch_size, _, n_blocks = block_mask.kv_num_blocks.shape
        block_size = block_mask.BLOCK_SIZE[0]
        n_tokens = n_blocks * block_size

        x = x.expand(batch_size, n_tokens, n_tokens, -1)
        indices = BlockSparseTensor.block_mask_to_index(block_mask)

        physical = x[indices]
        return BlockSparseTensor(physical, block_mask)

    def _build_lookup_table(self):
        batch_size, _, n_blocks = self.kv_num_blocks.shape
        kv_num_blocks_no_heads = self.kv_num_blocks[:, 0, :].flatten()
        total_num_blocks = torch.sum(kv_num_blocks_no_heads)

        bq_lookup = torch.nn.functional.pad(
            torch.cumsum(kv_num_blocks_no_heads.flatten()[:-1], dim=0),
            (1, 0)
        )
        bq_lookup = bq_lookup.reshape(batch_size, n_blocks, 1)

        k_impact = torch.argsort(self.kv_indices[:, 0, :, :], dim=-1)

        lookup_table = bq_lookup + k_impact
        lookup_table = torch.clip(lookup_table, max=total_num_blocks-1)
        return lookup_table


    def __getitem__(self, index):
        b, q, k, c = index
        W = self.block_size
        return self.physical[self.lookup_table[b, q//W, k//W], q%W, k%W, c]

    def _unwrap(self, other):
        if isinstance(other, BlockSparseTensor):
            return other.physical
        return other

    def _wrap(self, tensor):
        return BlockSparseTensor(tensor, self.block_mask)

    def __add__(self, other):
        return self._wrap(self.physical + self._unwrap(other))

    def __radd__(self, other):
        return self._wrap(self._unwrap(other) + self.physical)

    def __sub__(self, other):
        return self._wrap(self.physical - self._unwrap(other))

    def __rsub__(self, other):
        return self._wrap(self._unwrap(other) - self.physical)

    def __mul__(self, other):
        return self._wrap(self.physical * self._unwrap(other))

    def __rmul__(self, other):
        return self._wrap(self._unwrap(other) * self.physical)

    def __truediv__(self, other):
        return self._wrap(self.physical / self._unwrap(other))

    def __rtruediv__(self, other):
        return self._wrap(self._unwrap(other) / self.physical)

    def __pow__(self, other):
        return self._wrap(self.physical ** self._unwrap(other))

    def __neg__(self):
        return self._wrap(-self.physical)

    def __eq__(self, other):
        return self._wrap(self.physical == self._unwrap(other))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def unwrap(x):
            return x.physical if isinstance(x, BlockSparseTensor) else x

        unwrapped_args = tuple(unwrap(a) for a in args)
        unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        result = func(*unwrapped_args, **unwrapped_kwargs)

        if isinstance(result, torch.Tensor):
            return self._wrap(result)
        elif isinstance(result, tuple):
            return tuple(self._wrap(r) if isinstance(r, torch.Tensor) else r for r in result)
        else:
            return result

    def clone(self):
        return self._wrap(self.physical.clone())

    def detach(self):
        return self._wrap(self.physical.detach())

    def requires_grad_(self, requires_grad=True):
        self.physical.requires_grad_(requires_grad)
        return self

    def to(self, *args, **kwargs):
        return self._wrap(self.physical.to(*args, **kwargs))

    @property
    def device(self):
        return self.physical.device

    @property
    def dtype(self):
        return self.physical.dtype

    def __repr__(self):
        return f"BlockSparseTensor(shape={tuple(self.physical.shape)}, device={self.device})"

