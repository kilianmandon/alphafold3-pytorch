import torch
import utils


class LayoutConversion:
    def __init__(self, gather_inds, target_mask, n_layout_dims):
        self.gather_inds = gather_inds
        self.target_mask = target_mask
        self.n_layout_dims = n_layout_dims
        self.feat_dims = {
            'positions': 1,
            'mask': 0,
            'element': 0,
            'atom_name_chars': 1,
            'ref_space_uid': 0,
        }

    def __call__(self, feat, n_feat_dims):
        return self.apply(feat, n_feat_dims)

    def apply(self, feat: torch.Tensor, n_feat_dims: int):
        batch_shape = feat.shape[:-n_feat_dims-self.n_layout_dims]
        layout_shape = feat.shape[-n_feat_dims-self.n_layout_dims:-n_feat_dims]

        if n_feat_dims > 0:
            feat_dim_shape = feat.shape[-n_feat_dims:]
        else:
            feat_dim_shape = tuple()

        feat_flat = feat.reshape(batch_shape+(-1,)+feat_dim_shape)

        gathered = utils.batched_gather(
            feat_flat, self.gather_inds, batch_shape)
        target_mask_bc_shape = self.target_mask.shape + (1,) * n_feat_dims
        gathered = gathered * self.target_mask.reshape(target_mask_bc_shape)
        return gathered

    def apply_dict(self, feat_dict):
        return {
            key: self.apply(feat, self.feat_dims[key]) for key, feat in feat_dict.items()
        }

    @staticmethod
    def stack(layout_conversions):
        all_gather_inds = [lc.gather_inds for lc in layout_conversions]
        all_target_masks = [lc.target_mask for lc in layout_conversions]
        n_layout_dims = [lc.n_layout_dims for lc in layout_conversions]

        assert len(set(n_layout_dims)) == 1
        n_layout_dims = n_layout_dims[0]

        return LayoutConversion(
            torch.stack(all_gather_inds, dim=0),
            torch.stack(all_target_masks, dim=0),
            n_layout_dims
        )


class AtomLayout:
    def __init__(self, 
                 tokens_to_queries: LayoutConversion,
                 tokens_to_keys: LayoutConversion, 
                 queries_to_keys: LayoutConversion, 
                 queries_to_tokens: LayoutConversion, 
                 pure_tokens_to_queries: LayoutConversion, 
                 pure_tokens_to_keys: LayoutConversion, 
                 pure_pair_to_qk: LayoutConversion):

        self.tokens_to_queries = tokens_to_queries
        self.tokens_to_keys = tokens_to_keys
        self.queries_to_keys = queries_to_keys
        self.queries_to_tokens = queries_to_tokens
        self.pure_pair_to_qk = pure_pair_to_qk
        self.pure_tokens_to_queries = pure_tokens_to_queries
        self.pure_tokens_to_keys = pure_tokens_to_keys

    @staticmethod
    def stack(atom_layouts):
        t2q = LayoutConversion.stack(
            al.tokens_to_queries for al in atom_layouts)
        t2k = LayoutConversion.stack(al.tokens_to_keys for al in atom_layouts)
        q2k = LayoutConversion.stack(al.queries_to_keys for al in atom_layouts)
        q2t = LayoutConversion.stack(al.queries_to_tokens for al in atom_layouts)
        pt2q = LayoutConversion.stack(al.pure_tokens_to_queries for al in atom_layouts)
        pt2k = LayoutConversion.stack(al.pure_tokens_to_keys for al in atom_layouts)
        pp2qk = LayoutConversion.stack(al.pair_to_qk for al in atom_layouts)
        return AtomLayout(t2q, t2k, q2k, q2t, pt2q, pt2k, pp2qk)

    @staticmethod
    def calculate_key_inds(atom_count, padded_atom_count):
        centers = torch.arange(16, padded_atom_count, 32)
        bounds = torch.stack((centers-64, centers+64), dim=1)

        exceeding_left = bounds[:, 0] < 0
        exceeding_right = bounds[:, 1] > atom_count
        bounds[exceeding_left] -= bounds[exceeding_left, :1]
        bounds[exceeding_right] -= bounds[exceeding_right, 1:] - atom_count

        inds = torch.stack([torch.arange(a[0], a[1]) for a in bounds], dim=0)
        return inds

    @staticmethod
    def from_single_mask(mask):
        # mask has shape (N_tokens, 24)
        N_tokens = mask.shape[0]
        N_blocks = N_tokens * 24 // 32

        mask_flat = mask.flatten()
        nonzero = torch.nonzero(mask_flat).squeeze()
        nonzero_mask = torch.ones(nonzero.shape)
        zero = torch.nonzero(mask_flat == 0).squeeze()
        zero_mask = torch.zeros(zero.shape)

        tokens_to_flat = torch.cat((nonzero, zero))
        flat_mask = torch.cat((nonzero_mask, zero_mask))

        flat_to_tokens = torch.argsort(tokens_to_flat)
        flat_to_tokens_mask = mask_flat

        flat_to_keys = AtomLayout.calculate_key_inds(
            nonzero.numel(), mask_flat.numel())

        tokens_to_queries = tokens_to_flat.reshape(N_blocks, 32)
        tokens_to_keys = tokens_to_flat[flat_to_keys]
        queries_to_keys = flat_to_keys
        queries_to_tokens = flat_to_tokens.reshape(N_tokens, 24)

        pure_tokens_to_queries = tokens_to_queries//24
        pure_tokens_to_keys = tokens_to_keys//24
        pure_pair_to_qk = pure_tokens_to_queries[:, :,
                                                 None] * N_tokens + pure_tokens_to_keys[:, None, :]

        token_mask = mask
        queries_mask = flat_mask.reshape(N_blocks, 32)
        keys_mask = torch.ones((N_blocks, 128))
        qk_mask = queries_mask[:, :, None] * keys_mask[:, None, :]

        t2q = LayoutConversion(tokens_to_queries, queries_mask, 2)
        t2k = LayoutConversion(tokens_to_keys, keys_mask, 2)
        q2k = LayoutConversion(queries_to_keys, keys_mask, 2)
        q2t = LayoutConversion(queries_to_tokens, token_mask, 2)

        pt2q = LayoutConversion(pure_tokens_to_queries, queries_mask, 1)
        pt2k = LayoutConversion(pure_tokens_to_keys, keys_mask, 1)
        pp2qk = LayoutConversion(pure_pair_to_qk, qk_mask, 2)

        return AtomLayout(t2q, t2k, q2k, q2t, pt2q, pt2k, pp2qk)
