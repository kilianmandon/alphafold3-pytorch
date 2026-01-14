import numpy as np
import torch

from atom_layout import LayoutConversion, AtomLayout


def pad_to_shape_np(data, padded_shape, value=0):
    padded = np.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)
    inds = tuple(slice(i) for i in data.shape)
    padded[inds] = data
    return padded


def pad_to_shape(data, padded_shape, value=0):
    padded = torch.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)
    inds = tuple(slice(i) for i in data.shape)
    padded[inds] = data
    return padded

def crop_pad_to_shape(data, padded_shape, value=0):
    inds = tuple(slice(min(i, j)) for i, j in zip(data.shape, padded_shape))
    data = data[inds]
    return pad_to_shape(data, padded_shape, value)

def round_down_to(data, rounding_target, return_indices=False):
    sorting_indices = np.argsort(rounding_target)[::-1]
    target_inds = np.argmax(rounding_target[sorting_indices] <= data[..., None], axis=-1)
    target_inds = sorting_indices[target_inds]

    if return_indices:
        return rounding_target[target_inds], target_inds
    else:
        return rounding_target[target_inds]

def round_up_to(data, rounding_target, return_indices=False):
    data = np.array(data)
    sorting_indices = np.argsort(rounding_target)
    target_inds = np.argmax(rounding_target[sorting_indices] >= data[..., None], axis=-1)
    target_inds = sorting_indices[target_inds]

    if return_indices:
        return rounding_target[target_inds], target_inds
    else:
        return rounding_target[target_inds]


def batched_gather(feat, gather_inds, batch_shape):
    # feat has shape (**batch_shape, gather_dim, **feat_dims)
    # gather_inds has shape (**batch_shape, **gather_shape)
    # out has shape (**batch_shape, **gather_shape, **feat_dims)
    feat_dims = feat.shape[len(batch_shape)+1:]
    gather_shape = gather_inds.shape[len(batch_shape):]
    out_shape = batch_shape + gather_shape + feat_dims
    gather_inds = gather_inds.flatten(-len(gather_shape))
    gather_inds_pre_bc_shape = gather_inds.shape + (1,) * len(feat_dims)
    gather_inds_post_bc_shape = gather_inds.shape + feat_dims
    gather_inds = gather_inds.reshape(gather_inds_pre_bc_shape).broadcast_to(gather_inds_post_bc_shape)

    feat_gathered = torch.gather(feat, len(batch_shape), gather_inds)
    return feat_gathered.reshape(out_shape)

def masked_mean_np(feat, mask, axis, keepdims=False):
    feat_sum = np.sum(feat * mask, axis=axis, keepdims=keepdims)
    count = np.sum(mask, axis=axis, keepdims=keepdims)
    return feat_sum / np.clip(count, a_min=1e-10, a_max=None)

def masked_mean(feat, mask, dim, keepdim=False):
    feat_sum = (feat*mask).sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim)
    return feat_sum / torch.clip(count, min=1e-10)



def quat_mul(q1, q2):
    """
    Batched multiplication of two quaternions.

    Args:
        q1 (torch.tensor): Quaternion of shape (*, 4).
        q2 (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Quaternion of shape (*, 4).
    """
    
    a1 = q1[...,0:1] # a1 has shape (*, 1)
    v1 = q1[..., 1:] # v1 has shape (*, 3)
    a2 = q2[...,0:1] # a2 has shape (*, 1)
    v2 = q2[..., 1:] # v2 has shape (*, 3)

    q_out = None

    a_out = a1*a2 - torch.sum(v1*v2, dim=-1, keepdim=True)
    v_out = a1*v2 + a2*v1 + torch.linalg.cross(v1, v2, dim=-1)

    q_out = torch.cat((a_out, v_out), dim=-1)

    return q_out

def conjugate_quat(q):
    """
    Calculates the conjugate of a quaternion, i.e. 
    (a, -v) for q=(a, v).

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).

    Returns:
        torch.tensor: Conjugate quaternion of shape (*, 4).
    """

    q_out = None
    
    q_out = q.clone()
    q_out[..., 1:] = -q_out[..., 1:]

    return q_out

def quat_vector_mul(q, v):
    """
    Rotates a vector by a quaternion according to q*v*q', where q' 
    denotes the conjugate. The vector v is promoted to a quaternion 
    by padding a 0 for the scalar aprt.

    Args:
        q (torch.tensor): Quaternion of shape (*, 4).
        v (torch.tensor): Vector of shape (*, 3).

    Returns:
        torch.tensor: Rotated vector of shape (*, 3).
    """
    batch_shape = v.shape[:-1]
    v_out = None

    zero_pad = torch.zeros(batch_shape+(1,), device=v.device, dtype=v.dtype)
    padded_v = torch.cat((zero_pad, v), dim=-1)

    q_out = quat_mul(q, quat_mul(padded_v, conjugate_quat(q)))
    v_out = q_out[...,1:]

    return v_out

def move_to_device(obj, device=None, dtype=None):
    """Recursively move all tensors in a nested structure to a specified device."""
    if isinstance(obj, torch.Tensor):  
        return obj.to(device=device, dtype=dtype)
    elif isinstance(obj, dict):  
        return {key: move_to_device(value, device, dtype) for key, value in obj.items()}  
    elif isinstance(obj, list):  
        return [move_to_device(item, device, dtype) for item in obj]  
    elif isinstance(obj, tuple):  
        return tuple(move_to_device(item, device) for item in obj)  
    elif isinstance(obj, AtomLayout) or isinstance(obj, LayoutConversion):
        for attr in vars(obj):
            setattr(obj, attr, move_to_device(getattr(obj, attr), device, dtype))
        return obj
    else:
        return obj