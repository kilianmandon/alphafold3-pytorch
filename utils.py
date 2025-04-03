import io
import modelcif
import modelcif.model
import modelcif.dumper
import torch


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


def masked_mean(feat, mask, dim, keepdim=False):
    feat_sum = (feat*mask).sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim)
    return feat_sum / torch.clip(count, min=1e-10)

def to_modelcif(token_positions, token_mask, inp, ccd):
    all_asym_units = []
    all_atom_iterators = []

    system = modelcif.System(title='AlphaFold 3 Prediction')
    chunks = [seq.tokenCount for seq in inp.sequences]
    totalCount = sum(chunks)
    
    split_positions = torch.split(token_positions[:totalCount], chunks)
    split_mask = torch.split(token_mask[:totalCount], chunks)

    for i, seq in enumerate(inp.sequences):
        asym_units, atom_iterator = seq.get_model(ccd, split_positions[i], split_mask[i])
        all_asym_units += asym_units
        all_atom_iterators.append(atom_iterator)
    modeled_assembly = modelcif.Assembly(all_asym_units, name='Modeled Assembly')

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            for iterator in all_atom_iterators:
                yield from iterator

    model = _MyModel(assembly=modeled_assembly, name='Model')
    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)
    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])

    return fh.getvalue()



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