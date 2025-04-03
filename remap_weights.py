import copy
import struct
import sys
import numpy as np
import jax.numpy as jnp
import torch
import tqdm
import zstandard as zstd

def create_atom_att_encoder(jax_base, jax_sub, pytorch_base, use_trunk=False):
    name_map_att_enc = {
        f'{jax_base}/{jax_sub}_embed_ref_pos': {
            'weights': f'{pytorch_base}.embed_ref_pos.weight'
        },
        f'{jax_base}/{jax_sub}_embed_ref_charge': {
            'weights': f'{pytorch_base}.embed_ref_charge.weight'
        },
        f'{jax_base}/{jax_sub}_embed_ref_element': {
            'weights': f'{pytorch_base}.embed_ref_element.weight'
        },
        f'{jax_base}/{jax_sub}_embed_ref_mask': {
            'weights': f'{pytorch_base}.embed_ref_mask.weight'
        },
        f'{jax_base}/{jax_sub}_embed_ref_atom_name': {
            'weights': f'{pytorch_base}.embed_ref_atom_name.weight'
        },
        f'{jax_base}/{jax_sub}_single_to_pair_cond_col_1': {
            'weights': f'{pytorch_base}.single_to_pair_col.weight'
        },
        f'{jax_base}/{jax_sub}_single_to_pair_cond_row_1': {
            'weights': f'{pytorch_base}.single_to_pair_row.weight'
        },
        f'{jax_base}/{jax_sub}_embed_pair_offsets_1': {
            'weights': f'{pytorch_base}.embed_pair_offsets.weight'
        },
        f'{jax_base}/{jax_sub}_embed_pair_distances_1': {
            'weights': f'{pytorch_base}.embed_pair_distances.weight'
        },
        f'{jax_base}/{jax_sub}_embed_pair_offsets_valid': {
            'weights': f'{pytorch_base}.embed_pair_mask.weight'
        },
        f'{jax_base}/{jax_sub}_pair_mlp_1': {
            'weights': f'{pytorch_base}.pair_mlp.1.weight'
        },
        f'{jax_base}/{jax_sub}_pair_mlp_2': {
            'weights': f'{pytorch_base}.pair_mlp.3.weight'
        },
        f'{jax_base}/{jax_sub}_pair_mlp_3': {
            'weights': f'{pytorch_base}.pair_mlp.5.weight'
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/pair_input_layer_norm': {
            'split': 'XXX',
            'scale': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_z.weight'
            # 'scale': f'{pytorch_base}.atom_transformer.pair_input_layer_norm.weight'
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/pair_logits_projection': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_b.weight',
            # 'flatten': (-2, -1),
            # 'weights': f'{pytorch_base}.atom_transformer.pair_logits_projection.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderqsingle_cond_layer_norm': {
            'split': 'XXX',
            'scale': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_layer_norm.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderksingle_cond_layer_norm': {
            'split': 'XXX',
            'scale': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_layer_norm.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderqsingle_cond_scale': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_scale.weight',
            'bias': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_scale.bias',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderksingle_cond_scale': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_scale.weight',
            'bias': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_scale.bias',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderqsingle_cond_bias': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_bias.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderksingle_cond_bias': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_bias.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderq_projection': {
            'split': 'XXX',
            'flatten': (-2, -1),
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_q.weight',
            'bias': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_q.bias',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderk_projection': {
            'split': 'XXX',
            'flatten': (-2, -1),
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_k.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderv_projection': {
            'split': 'XXX',
            'flatten': (-2, -1),
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_v.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encodergating_query': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.linear_g.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encodertransition2': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_transition.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderadaptive_zero_cond': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_cond.weight',
            'bias': f'{pytorch_base}.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_cond.bias',
        },

        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_single_cond_layer_norm': {
            'split': 'XXX',
            'scale': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_layer_norm.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_single_cond_scale': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_scale.weight',
            'bias': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_scale.bias',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_single_cond_bias': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_bias.weight',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_transition1#X': {
            'split': 'XXX',
            'index': (slice(None), slice(256)),
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.linear_a1.weight'
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_transition1#Y': {
            'split': 'XXX',
            'index': (slice(None), slice(256, None)),
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.linear_a2.weight'
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_adaptive_zero_cond': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_cond.weight',
            'bias': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_cond.bias',
        },
        f'{jax_base}/{jax_sub}_atom_transformer_encoder/__layer_stack_with_per_layer/{jax_sub}_atom_transformer_encoderffw_transition2': {
            'split': 'XXX',
            'weights': f'{pytorch_base}.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_transition.weight',
        },
        f'{jax_base}/{jax_sub}_project_atom_features_for_aggr': {
            'weights': f'{pytorch_base}.project_atom_features.weight',
        }
    }

    if use_trunk:
        name_map_add = {
            f'{jax_base}/{jax_sub}_embed_trunk_single_cond': {
                'weights': f'{pytorch_base}.trunk_linear_s.weight',
            },
            f'{jax_base}/{jax_sub}_lnorm_trunk_single_cond': {
                'scale': f'{pytorch_base}.trunk_layer_norm_s.weight',
            },
            f'{jax_base}/{jax_sub}_embed_trunk_pair_cond': {
                'weights': f'{pytorch_base}.trunk_linear_z.weight',
            },
            f'{jax_base}/{jax_sub}_lnorm_trunk_pair_cond': {
                'scale': f'{pytorch_base}.trunk_layer_norm_z.weight',
            },
            f'{jax_base}/{jax_sub}_atom_positions_to_features': {
                'weights': f'{pytorch_base}.trunk_linear_r.weight',
            },
        }
        name_map_att_enc.update(name_map_add)

    return name_map_att_enc

name_map_input_embedder = {
    'diffuser/evoformer/left_single': {
        'weights': 'evoformer.input_embedder.left_single.weight',
    },
    'diffuser/evoformer/right_single': {
        'weights': 'evoformer.input_embedder.right_single.weight',
    },
    'diffuser/evoformer/bond_embedding': {
        'weights': 'evoformer.input_embedder.bond_embedding.weight',
    },
    'diffuser/evoformer/~_relative_encoding/position_activations': {
        'weights': 'evoformer.input_embedder.position_activations.weight',
    },
    'diffuser/evoformer/single_activations': {
        'weights': 'evoformer.input_embedder.single_embedding.weight',
    },
}

name_map_evoformer = {
    'diffuser/evoformer/prev_embedding_layer_norm': {
        'scale': 'evoformer.layer_norm_prev_z.weight',
        'offset': 'evoformer.layer_norm_prev_z.bias',
    },
    'diffuser/evoformer/prev_embedding': {
        'weights': 'evoformer.prev_z_embedding.weight',
    },
    'diffuser/evoformer/prev_single_embedding_layer_norm': {
        'scale': 'evoformer.layer_norm_prev_s.weight',
        'offset': 'evoformer.layer_norm_prev_s.bias',
    },
    'diffuser/evoformer/prev_single_embedding': {
        'weights': 'evoformer.prev_s_embedding.weight',
    },
}

name_map_msa_module = {
    'diffuser/evoformer/msa_activations': {
        'weights': 'evoformer.msa_module.linear_m.weight',
    },
    'diffuser/evoformer/extra_msa_target_feat': {
        'weights': 'evoformer.msa_module.linear_s.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean#X': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'output_b': 'evoformer.msa_module.blocks.XXX.opm.linear_out.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean#Y': {
        'split': 'XXX',
        'flatten': (-3, -2),
        # torch.Size([4, 32, 32, 128])
        'output_w': 'evoformer.msa_module.blocks.XXX.opm.linear_out.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean/layer_norm_input': {
        'split': 'XXX',
        # torch.Size([4, 64])
        'scale': 'evoformer.msa_module.blocks.XXX.opm.layer_norm.weight',
        # torch.Size([4, 64])
        'offset': 'evoformer.msa_module.blocks.XXX.opm.layer_norm.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean/left_projection': {
        'split': 'XXX',
        # torch.Size([4, 64, 32])
        'weights': 'evoformer.msa_module.blocks.XXX.opm.linear_a.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean/right_projection': {
        'split': 'XXX',
        # torch.Size([4, 64, 32])
        'weights': 'evoformer.msa_module.blocks.XXX.opm.linear_b.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/pair_logits': {
        'split': 'XXX',
        # torch.Size([4, 128, 8])
        'weights': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.linear_b.weight'
    },


    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/act_norm': {
        'split': 'XXX',
        # torch.Size([4, 64])
        'offset': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.layer_norm_m.bias',
        # torch.Size([4, 64])
        'scale': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.layer_norm_m.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/pair_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/gating_query': {
        'split': 'XXX',
        # torch.Size([4, 64, 64])
        'weights': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 64, 64])
        'weights': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.linear_out.weight'
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_attention1/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # torch.Size([4, 64, 8, 8])
        'weights': 'evoformer.msa_module.blocks.XXX.msa_pair_weighted.linear_v.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_transition/transition1#X': {
        'split': 'XXX',
        'index': (slice(None), slice(None, 256)),
        # torch.Size([4, 64, 512])
        'weights': 'evoformer.msa_module.blocks.XXX.transition.linear_a.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_transition/transition1#Y': {
        'split': 'XXX',
        'index': (slice(None), slice(256, None)),
        # torch.Size([4, 64, 512])
        'weights': 'evoformer.msa_module.blocks.XXX.transition.linear_b.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_transition/input_layer_norm': {
        'split': 'XXX',
        # torch.Size([4, 64])
        'scale': 'evoformer.msa_module.blocks.XXX.transition.layer_norm.weight',
        # torch.Size([4, 64])
        'offset': 'evoformer.msa_module.blocks.XXX.transition.layer_norm.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/msa_transition/transition2': {
        'split': 'XXX',
        'weights': 'evoformer.msa_module.blocks.XXX.transition.linear_out.weight',
    }
}

name_map_pair_stack = {
    # Mapping for triangle_multiplication_outgoing
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/left_norm_input': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.layer_norm_z.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/center_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.layer_norm_out.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.layer_norm_out.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/projection#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_a2.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/projection#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_b2.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/gate#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_a1.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/gate#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_b1.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/gating_linear': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_g.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_outgoing/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_outgoing.linear_out.weight',
    },

    # Mapping for triangle_multiplication_incoming
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/left_norm_input': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.layer_norm_z.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/center_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.layer_norm_out.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.layer_norm_out.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/projection#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_a2.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/projection#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_b2.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/gate#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_a1.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/gate#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_b1.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/gating_linear': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_g.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/triangle_multiplication_incoming/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_mult_incoming.linear_out.weight',
    },

    # Mapping for pair_attention components (triangle_att_starting in PyTorch)
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/act_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.layer_norm_z.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_q.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_k.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # torch.Size([4, 128, 4, 32]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_v.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/pair_bias_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 4])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_b.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/gating_query': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'transpose': False,
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_g.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention1/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_starting.linear_out.weight',
    },

    # Mapping for pair_attention2 components (triangle_att_ending in PyTorch)
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/act_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.layer_norm_z.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_q.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_k.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # torch.Size([4, 128, 4, 32]) -> need to reshape
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_v.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/pair_bias_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 4])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_b.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/gating_query': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'transpose': False,
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_g.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_attention2/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.triangle_att_ending.linear_out.weight',
    },

    # Mapping for pair_transition components
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_transition/input_layer_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.msa_module.blocks.XXX.core.transition.layer_norm.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.msa_module.blocks.XXX.core.transition.layer_norm.bias',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_transition/transition1#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 1024])
        'weights': 'evoformer.msa_module.blocks.XXX.core.transition.linear_a.weight',
        'index': (slice(None), slice(512)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_transition/transition1#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 1024])
        'weights': 'evoformer.msa_module.blocks.XXX.core.transition.linear_b.weight',
        'index': (slice(None), slice(512, None)),
    },
    'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/pair_transition/transition2': {
        'split': 'XXX',
        # torch.Size([4, 512, 128])
        'weights': 'evoformer.msa_module.blocks.XXX.core.transition.linear_out.weight',
    }
}

name_map_pairformer = {
    # Triangle Multiplication Outgoing
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/left_norm_input': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.layer_norm_z.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/center_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.layer_norm_out.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.layer_norm_out.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/projection#1': {
        'split': 'XXX',
        'index': (slice(None), slice(None, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_a2.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/projection#2': {
        'split': 'XXX',
        'index': (slice(None), slice(1, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_b2.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/gate#1': {
        'split': 'XXX',
        'index': (slice(None), slice(None, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_a1.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/gate#2': {
        'split': 'XXX',
        'index': (slice(None), slice(1, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_b1.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/gating_linear': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/output_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_outgoing.linear_out.weight',
    },

    # Triangle Multiplication Incoming
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/left_norm_input': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.layer_norm_z.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/center_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.layer_norm_out.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.layer_norm_out.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/projection#1': {
        'split': 'XXX',
        'index': (slice(None), slice(None, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_a2.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/projection#2': {
        'split': 'XXX',
        'index': (slice(None), slice(1, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_b2.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/gate#1': {
        'split': 'XXX',
        'index': (slice(None), slice(None, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_a1.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/gate#2': {
        'split': 'XXX',
        'index': (slice(None), slice(1, None, 2)),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_b1.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/gating_linear': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/output_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_mult_incoming.linear_out.weight',
    },

    # Pair Attention 1 (Triangle Attention Starting)
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/act_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.layer_norm_z.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_q.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_k.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_v.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/pair_bias_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_b.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/gating_query': {
        'split': 'XXX',
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/output_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_starting.linear_out.weight',
    },

    # Pair Attention 2 (Triangle Attention Ending)
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/act_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.layer_norm_z.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_q.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_k.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_v.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/pair_bias_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_b.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/gating_query': {
        'split': 'XXX',
        'transpose': False,
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/output_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.triangle_att_ending.linear_out.weight',
    },

    # Transition for Core
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/input_layer_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.core.transition.layer_norm.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.core.transition.layer_norm.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/transition1#X': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.transition.linear_a.weight',
        'index': (slice(None), slice(512)),
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/transition1#Y': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.transition.linear_b.weight',
        'index': (slice(None), slice(512, None)),
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/transition2': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.core.transition.linear_out.weight',
    },

    # Attention Pair Bias
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_layer_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.att_pair_bias.layer_norm_a.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.att_pair_bias.layer_norm_a.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_pair_logits_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.att_pair_bias.layer_norm_z.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.att_pair_bias.layer_norm_z.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_q_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_q.weight',
        'bias': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_q.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_k_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_k.weight',
    },
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_v.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_pair_logits_projection': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_b.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_gating_query': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_g.weight',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_transition2': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.att_pair_bias.linear_out.weight',
    },


    # Transition (Outside Core)
    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/input_layer_norm': {
        'split': 'XXX',
        'scale': 'evoformer.pairformer.blocks.XXX.single_transition.layer_norm.weight',
        'offset': 'evoformer.pairformer.blocks.XXX.single_transition.layer_norm.bias',
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/transition1#X': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.single_transition.linear_a.weight',
        'index': (slice(None), slice(1536)),
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/transition1#Y': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.single_transition.linear_b.weight',
        'index': (slice(None), slice(1536, None)),
    },

    'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/transition2': {
        'split': 'XXX',
        'weights': 'evoformer.pairformer.blocks.XXX.single_transition.linear_out.weight',
    },
}

name_map_template_pair_stack = {
    # Mapping for triangle_multiplication_outgoing
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/left_norm_input': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.layer_norm_z.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/center_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.layer_norm_out.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.layer_norm_out.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/projection#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_a2.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/projection#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_b2.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/gate#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_a1.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/gate#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_b1.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/gating_linear': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_g.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_outgoing/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_outgoing.linear_out.weight',
    },

    # Mapping for triangle_multiplication_incoming
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/left_norm_input': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.layer_norm_z.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/center_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.layer_norm_out.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.layer_norm_out.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/projection#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_a2.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/projection#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_b2.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/gate#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_a1.weight',
        'index': (slice(None), slice(None, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/gate#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 256])
        # Maps to a1, a2, b1, b2 linear layers combined
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_b1.weight',
        'index': (slice(None), slice(1, None, 2)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/gating_linear': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_g.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/triangle_multiplication_incoming/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_mult_incoming.linear_out.weight',
    },

    # Mapping for pair_attention components (triangle_att_starting in PyTorch)
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/act_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.layer_norm_z.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_q.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_k.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # torch.Size([4, 128, 4, 32]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_v.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/pair_bias_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 4])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_b.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/gating_query': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'transpose': False,
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_g.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention1/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_starting.linear_out.weight',
    },

    # Mapping for pair_attention2 components (triangle_att_ending in PyTorch)
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/act_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.layer_norm_z.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.layer_norm_z.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/q_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_q.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/k_projection': {
        'split': 'XXX',
        'flatten': (-3, -2),
        'transpose': False,
        # torch.Size([4, 4, 32, 128]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_k.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/v_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # torch.Size([4, 128, 4, 32]) -> need to reshape
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_v.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/pair_bias_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 4])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_b.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/gating_query': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'transpose': False,
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_g.weight',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_attention2/output_projection': {
        'split': 'XXX',
        # torch.Size([4, 128, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.triangle_att_ending.linear_out.weight',
    },

    # Mapping for pair_transition components
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_transition/input_layer_norm': {
        'split': 'XXX',
        # torch.Size([4, 128])
        'scale': 'evoformer.template_embedder.pair_stack.XXX.transition.layer_norm.weight',
        # torch.Size([4, 128])
        'offset': 'evoformer.template_embedder.pair_stack.XXX.transition.layer_norm.bias',
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_transition/transition1#X': {
        'split': 'XXX',
        # torch.Size([4, 128, 1024])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.transition.linear_a.weight',
        'index': (slice(None), slice(128)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_transition/transition1#Y': {
        'split': 'XXX',
        # torch.Size([4, 128, 1024])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.transition.linear_b.weight',
        'index': (slice(None), slice(128, None)),
    },
    'diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/pair_transition/transition2': {
        'split': 'XXX',
        # torch.Size([4, 512, 128])
        'weights': 'evoformer.template_embedder.pair_stack.XXX.transition.linear_out.weight',
    }
}

name_map_template_non_stack = {
    # evoformer.template_embedder.linear_a.weight: torch.Size([64, 106])
    # evoformer.template_embedder.linear_z.weight: torch.Size([64, 128])
    # evoformer.template_embedder.layer_norm_z.weight: torch.Size([128])
    # evoformer.template_embedder.layer_norm_z.bias: torch.Size([128])
    # evoformer.template_embedder.layer_norm_v.weight: torch.Size([64])
    # evoformer.template_embedder.layer_norm_v.bias: torch.Size([64])
    # evoformer.template_embedder.linear_out.weight: torch.Size([128, 64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_0': {
        'weights': 'evoformer.template_embedder.linear_a.weight#0',
    },
    #   weights: torch.Size([39, 64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_1': {
        'weights': 'evoformer.template_embedder.linear_a.weight#1',
        'index': (None, slice(None)),
    },
    #   weights: torch.Size([64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_2': {
        'weights': 'evoformer.template_embedder.linear_a.weight#2',
    },
    #   weights: torch.Size([31, 64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_3': {
        'weights': 'evoformer.template_embedder.linear_a.weight#3',
    },
    #   weights: torch.Size([31, 64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_4': {
        'weights': 'evoformer.template_embedder.linear_a.weight#4',
        'index': (None, slice(None)),
    },
    #   weights: torch.Size([64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_5': {
        'weights': 'evoformer.template_embedder.linear_a.weight#5',
        'index': (None, slice(None)),
    },
    #     weights   torch.Size([64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_6': {
        'weights': 'evoformer.template_embedder.linear_a.weight#6',
        'index': (None, slice(None)),
    },
    #   weights: torch.Size([64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_7': {
        'weights': 'evoformer.template_embedder.linear_a.weight#7',
        'index': (None, slice(None)),
    },
    #   weights: torch.Size([64])
    'diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_8': {
        'weights': 'evoformer.template_embedder.linear_z.weight',
    },
    #   weights: torch.Size([128, 64])
    'diffuser/evoformer/template_embedding/single_template_embedding/query_embedding_norm': {
        'scale': 'evoformer.template_embedder.layer_norm_z.weight',
        'offset': 'evoformer.template_embedder.layer_norm_z.bias',
    },
    #   scale: torch.Size([128])
    #   offset: torch.Size([128])
    'diffuser/evoformer/template_embedding/single_template_embedding/output_layer_norm': {
        'scale': 'evoformer.template_embedder.layer_norm_v.weight',
        'offset': 'evoformer.template_embedder.layer_norm_v.bias',
    },
    #   offset: torch.Size([64])
    #   scale: torch.Size([64])
    'diffuser/evoformer/template_embedding/output_linear': {
        'weights': 'evoformer.template_embedder.linear_out.weight',
    }
}

name_map_diffusion_conditioning = {
    'diffuser/~/diffusion_head/pair_cond_initial_projection': {
        'weights': 'diffusion_module.diffusion_conditioning.linear_z.weight',
    },
    'diffuser/~/diffusion_head/pair_cond_initial_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.layer_norm_z.weight',
    },
    'diffuser/~/diffusion_head/single_cond_initial_projection': {
        'weights': 'diffusion_module.diffusion_conditioning.linear_s.weight',
    },
    'diffuser/~/diffusion_head/single_cond_initial_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.layer_norm_s.weight',
    },
    'diffuser/~/diffusion_head/noise_embedding_initial_projection': {
        'weights': 'diffusion_module.diffusion_conditioning.linear_fourier.weight',
    },
    'diffuser/~/diffusion_head/noise_embedding_initial_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.layer_norm_fourier.weight',
    },
    # Pair Transition
    'diffuser/~/diffusion_head/pair_transition_0ffw_transition1#X': {
        'index': (slice(None), slice(256)),
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.0.linear_a.weight'
    },
    'diffuser/~/diffusion_head/pair_transition_0ffw_transition1#Y': {
        'index': (slice(None), slice(256, None)),
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.0.linear_b.weight'
    },
    'diffuser/~/diffusion_head/pair_transition_0ffw_layer_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.z_transition.0.layer_norm.weight',
        'offset': 'diffusion_module.diffusion_conditioning.z_transition.0.layer_norm.bias',
    },
    'diffuser/~/diffusion_head/pair_transition_0ffw_transition2': {
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.0.linear_out.weight',
    },
    'diffuser/~/diffusion_head/pair_transition_1ffw_transition1#X': {
        'index': (slice(None), slice(256)),
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.1.linear_a.weight'
    },
    'diffuser/~/diffusion_head/pair_transition_1ffw_transition1#Y': {
        'index': (slice(None), slice(256, None)),
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.1.linear_b.weight'
    },
    'diffuser/~/diffusion_head/pair_transition_1ffw_layer_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.z_transition.1.layer_norm.weight',
        'offset': 'diffusion_module.diffusion_conditioning.z_transition.1.layer_norm.bias',
    },
    'diffuser/~/diffusion_head/pair_transition_1ffw_transition2': {
        'weights': 'diffusion_module.diffusion_conditioning.z_transition.1.linear_out.weight',
    },
    # Single Transition
    'diffuser/~/diffusion_head/single_transition_0ffw_transition1#X': {
        'index': (slice(None), slice(768)),
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.0.linear_a.weight'
    },
    'diffuser/~/diffusion_head/single_transition_0ffw_transition1#Y': {
        'index': (slice(None), slice(768, None)),
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.0.linear_b.weight'
    },
    'diffuser/~/diffusion_head/single_transition_0ffw_layer_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.s_transition.0.layer_norm.weight',
        'offset': 'diffusion_module.diffusion_conditioning.s_transition.0.layer_norm.bias',
    },
    'diffuser/~/diffusion_head/single_transition_0ffw_transition2': {
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.0.linear_out.weight',
    },

    'diffuser/~/diffusion_head/single_transition_1ffw_transition1#X': {
        'index': (slice(None), slice(768)),
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.1.linear_a.weight'
    },
    'diffuser/~/diffusion_head/single_transition_1ffw_transition1#Y': {
        'index': (slice(None), slice(768, None)),
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.1.linear_b.weight'
    },
    'diffuser/~/diffusion_head/single_transition_1ffw_layer_norm': {
        'scale': 'diffusion_module.diffusion_conditioning.s_transition.1.layer_norm.weight',
        'offset': 'diffusion_module.diffusion_conditioning.s_transition.1.layer_norm.bias',
    },
    'diffuser/~/diffusion_head/single_transition_1ffw_transition2': {
        'weights': 'diffusion_module.diffusion_conditioning.s_transition.1.linear_out.weight',
    },
}

name_map_diffusion_basic = {
    'diffuser/~/diffusion_head/single_cond_embedding_norm': {
        'scale': 'diffusion_module.layer_norm_s.weight',
    },
    'diffuser/~/diffusion_head/single_cond_embedding_projection': {
        'weights': 'diffusion_module.linear_s.weight',
    },
    'diffuser/~/diffusion_head/output_norm': {
        'scale': 'diffusion_module.layer_norm_a.weight',
    },
}



name_map_diffusion_transformer = {
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformersingle_cond_scale': {
        'split': 'XXX',
        # Shape: [24, 768]
        'bias': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.layer_norm_a.single_cond_scale.bias',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.layer_norm_a.single_cond_scale.weight',
    },

    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/pair_logits_projection': {
        'split': 'XXX',
        # Shape: [6, 128, 4, 16]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_b.weight',
    },

    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerk_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # Shape: [24, 768, 16, 48]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_k.weight',
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformersingle_cond_bias': {
        'split': 'XXX',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.layer_norm_a.single_cond_bias.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerv_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # Shape: [24, 768, 16, 48]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_v.weight'
    },

    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformertransition2': {
        'split': 'XXX',
        # Shape: [24, 768, 768]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_out.weight',
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformersingle_cond_layer_norm': {
        'split': 'XXX',
        # Shape: [24, 384]
        'scale': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.layer_norm_a.single_cond_layer_norm.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformeradaptive_zero_cond': {
        'split': 'XXX',
        # Shape: [24, 768]
        'bias': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_out_adaptive.bias',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_out_adaptive.weight'
    },

    'diffuser/~/diffusion_head/transformer/pair_input_layer_norm': {
        'split': 'XXX',
        # Shape: [128]
        'scale': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.layer_norm_z.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerq_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # Shape: [24, 16, 48]
        'bias': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_q.bias',
        # Shape: [24, 768, 16, 48]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_q.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformergating_query': {
        'split': 'XXX',
        # Shape: [24, 768, 768]
        'weights': 'diffusion_module.diffusion_transformer.att_pair_bias.XXX.linear_g.weight'
    },

    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_adaptive_zero_cond': {
        'split': 'XXX',
        # Shape: [24, 768]
        'bias': 'diffusion_module.diffusion_transformer.cond_trans.XXX.ada_zero_init.linear_cond.bias',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.ada_zero_init.linear_cond.weight',
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_single_cond_layer_norm': {
        'split': 'XXX',
        # Shape: [24, 384]
        'scale': 'diffusion_module.diffusion_transformer.cond_trans.XXX.adaptive_layernorm.single_cond_layer_norm.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_single_cond_scale': {
        'split': 'XXX',
        # Shape: [24, 768]
        'bias': 'diffusion_module.diffusion_transformer.cond_trans.XXX.adaptive_layernorm.single_cond_scale.bias',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.adaptive_layernorm.single_cond_scale.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_single_cond_bias': {
        'split': 'XXX',
        # Shape: [24, 384, 768]
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.adaptive_layernorm.single_cond_bias.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_transition1#X': {
        'split': 'XXX',
        # Shape: [24, 768, 3072]
        'index': (slice(None), slice(1536)),
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.linear_a1.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_transition1#Y': {
        'split': 'XXX',
        # Shape: [24, 768, 3072]
        'index': (slice(None), slice(1536, None)),
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.linear_a2.weight'
    },
    'diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_transition2': {
        'split': 'XXX',
        # Shape: [24, 1536, 768]
        'weights': 'diffusion_module.diffusion_transformer.cond_trans.XXX.ada_zero_init.linear_transition.weight',
    },
}

name_map_atom_att_dec = {
    # Token act embedding
    'diffuser/~/diffusion_head/diffusion_project_token_features_for_broadcast': {
        'weights': 'diffusion_module.atom_att_dec.linear_a.weight',
    },

    # Transformer Decoder
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/pair_input_layer_norm': {
        # Shape: (3, 16)
        'split': 'XXX',
        'scale': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_z.weight',
        # 'scale': 'diffusion_module.atom_att_dec.atom_transformer.pair_input_layer_norm.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/pair_logits_projection': {
        # Shape: (3, 16, 4)
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_b.weight',
        # 'flatten': (-2, -1),
        # 'weights': 'diffusion_module.atom_att_dec.atom_transformer.pair_logits_projection.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderqsingle_cond_layer_norm': {
        'split': 'XXX',
        'scale': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_layer_norm.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderqsingle_cond_scale': {
        'split': 'XXX',
        # Shape: [3, 128, 128]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_scale.weight',
        # Shape: [3, 128]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_scale.bias',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderqsingle_cond_bias': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_q.single_cond_bias.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderq_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        # Shape: [3, 128, 4, 32]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_q.weight',
        # Shape: [3, 4, 32]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_q.bias',
    },


    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderksingle_cond_layer_norm': {
        'split': 'XXX',
        'scale': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_layer_norm.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderksingle_cond_scale': {
        'split': 'XXX',
        # Shape: [3, 128, 128]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_scale.weight',
        # Shape: [3, 128]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_scale.bias',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderksingle_cond_bias': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.layer_norm_k.single_cond_bias.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderk_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_k.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderv_projection': {
        'split': 'XXX',
        'flatten': (-2, -1),
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_v.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decodergating_query': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.linear_g.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decodertransition2': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_transition.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderadaptive_zero_cond': {
        'split': 'XXX',
        # Shape: [3, 128, 128]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_cond.weight',
        # Shape: [3, 128]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.attn_blocks.XXX.ada_zero_init.linear_cond.bias',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_single_cond_layer_norm': {
        'split': 'XXX',
        'scale': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_layer_norm.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_single_cond_scale': {
        'split': 'XXX',
        # Shape: [3, 128]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_scale.bias',
        # Shape: [3, 128, 128]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_scale.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_single_cond_bias': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.adaptive_layernorm.single_cond_bias.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_transition1#X': {
        'split': 'XXX',
        # Shape: (3, 128, 512)
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.linear_a1.weight',
        'index': (slice(None), slice(256)),
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_transition1#Y': {
        'split': 'XXX',
        # Shape: (3, 128, 512)
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.linear_a2.weight',
        'index': (slice(None), slice(256, None)),
    },

    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_transition2': {
        'split': 'XXX',
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_transition.weight',
    },
    'diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer/diffusion_atom_transformer_decoderffw_adaptive_zero_cond': {
        'split': 'XXX',
        # Shape: [3, 128]
        'bias': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_cond.bias',
        # Shape: [3, 128, 128]
        'weights': 'diffusion_module.atom_att_dec.atom_transformer.transition_blocks.XXX.ada_zero_init.linear_cond.weight',
    },

    # After Transformer Decoder
    'diffuser/~/diffusion_head/diffusion_atom_features_layer_norm': {
        'scale': 'diffusion_module.atom_att_dec.layer_norm_q.weight',
    },

    'diffuser/~/diffusion_head/diffusion_atom_features_to_position_update': {
        'weights': 'diffusion_module.atom_att_dec.linear_out.weight',
    },
}


name_map_input_cross_att = create_atom_att_encoder('diffuser', 'evoformer_conditioning', 'evoformer.input_embedder.atom_cross_att')
name_map_diff_cross_att = create_atom_att_encoder('diffuser/~/diffusion_head', 'diffusion', 'diffusion_module.atom_att_enc', use_trunk=True)

global_name_map = {
    **name_map_input_cross_att,
    **name_map_input_embedder,
    **name_map_evoformer,
    **name_map_msa_module,
    **name_map_pair_stack,
    **name_map_pairformer,
    **name_map_template_pair_stack,
    **name_map_template_non_stack,
    **name_map_diffusion_conditioning,
    **name_map_diffusion_basic,
    **name_map_diff_cross_att,
    **name_map_diffusion_transformer,
    **name_map_atom_att_dec,
}

    
def preprocess_transformer(jax_params):
    transformer_keys = set(key.split('#')[0] for key in name_map_diffusion_transformer.keys())

    for key in transformer_keys:
        sub_params = jax_params[key]
        sub_params_to_split = [a for a in [
                'weights', 'offset', 'bias', 'scale', 'output_b', 'output_w'] if a in sub_params.keys()]

        for sub_key in sub_params_to_split:
            val = sub_params[sub_key]
            if 'pair_logits_projection' in key:
                val = val.permute(0,2,1,3)
            if 'pair_input_layer_norm' in key:
                # Note: These parameters are shared in the AF3 code, but not in the paper
                val = val.expand(6, 4, -1)
            val = val.flatten(start_dim=0, end_dim=1)
            sub_params[sub_key] = val

def preprocess_atom_transformer(jax_params):
    markers_layernorm = ['atom_transformer_decoder/pair_input_layer_norm', 'atom_transformer_encoder/pair_input_layer_norm']
    markers_projection = ['atom_transformer_decoder/pair_logits_projection', 'atom_transformer_encoder/pair_logits_projection']

    pair_projection_keys = set(key.split('#')[0] for key in global_name_map if any(m in key for m in markers_projection))
    pair_layernorm_keys = set(key.split('#')[0] for key in global_name_map if any(m in key for m in markers_layernorm))

    for key in pair_projection_keys:
        sub_params = jax_params[key]
        # New Order: N_blocks, c_in, N_heads
        sub_params['weights'] = sub_params['weights'].permute(1, 0, 2)
    
    for key in pair_layernorm_keys:
        sub_params = jax_params[key]
        # Note: These parameters are shared in the AF3 code, but not in the paper
        sub_params['scale'] = sub_params['scale'].expand(3, -1)





def preprocessing(name_map, jax_params):
    preprocess_transformer(jax_params)
    preprocess_atom_transformer(jax_params)

def post_processing(params):
    to_cat = [
        f'evoformer.template_embedder.linear_a.weight#{i}' for i in range(8)]
    to_cat = [params.pop(k) for k in to_cat]
    params['evoformer.template_embedder.linear_a.weight'] = torch.cat(
        to_cat, dim=-1)


def presplit_names(name_map, params):
    name_map = split_layer_stack(name_map, params)
    return name_map


def with_new_indexing(current_entry, indexing, prepend=False):
    current_indexing = copy.deepcopy(current_entry)
    current_indexing = current_entry.get('index', [])
    if not isinstance(current_indexing, list):
        current_indexing = [current_indexing]

    if not prepend:
        current_indexing.append(indexing)
    else:
        current_indexing.insert(0, indexing)

    return current_indexing


def apply_index(w, index_list):
    if not isinstance(index_list, list):
        index_list = [index_list]
    for index in index_list:
        w = w[index]
    return w

def rreplace(s, old, new, maxcount):
    li = s.rsplit(old, maxcount)
    return new.join(li)

def split_layer_stack(name_map, params):
    new_entries = dict()
    keys_to_remove = []
    for group, subs in name_map.items():
        real_group = group.split('#')[0]
        split = subs.pop('split', None)

        sub_params_to_split = [a for a in [
            'weights', 'offset', 'bias', 'scale', 'output_b', 'output_w'] if a in subs]

        n = params[real_group][sub_params_to_split[0]].shape[0]

        if split is not None:
            keys_to_remove.append(group)
            for i in range(n):

                new_entry = {
                    **subs,
                    'index': with_new_indexing(subs, (i, ...), prepend=True),
                }

                for key in sub_params_to_split:
                    new_entry[key] = new_entry[key].replace(split, str(i))

                new_entries[f'{group}#{i}'] = new_entry

    for key in keys_to_remove:
        name_map.pop(key)
    name_map.update(new_entries)

    return name_map


def add_fourier_params(weights):
    w_path = 'data/params/diff_fourier_weight.pt'
    w_key = 'diffusion_module.diffusion_conditioning.fourier_w'
    b_path = 'data/params/diff_fourier_bias.pt'
    b_key = 'diffusion_module.diffusion_conditioning.fourier_b'
    weights[w_key] = torch.load(w_path, weights_only=False)
    weights[b_key] = torch.load(b_path, weights_only=False)


def remap_params(weights):
    new_weights = dict()
    name_map = global_name_map
    preprocessing(name_map, weights)
    name_map = presplit_names(name_map, weights)
    for group, subs in name_map.items():
        # Remove optional tags
        group = group.split('#')[0]
        flatten = subs.pop('flatten', None)
        index = subs.pop('index', None)
        transpose = subs.pop('transpose', True)

        def process_weight(w):
            if index is not None:
                w = apply_index(w, index)
            if flatten is not None:
                w = w.flatten(start_dim=flatten[0], end_dim=flatten[1])

            if w.ndim == 2 and transpose:
                w = w.T
            return w

        for name, target in subs.items():
            weight = weights[group][name]
            new_w = process_weight(weight)
            new_weights[target] = new_w

    post_processing(new_weights)
    add_fourier_params(new_weights)

    return new_weights

def open_jax_params(filename, mode="rb"):
    """
    Open a file normally if uncompressed, or as a decompressed stream if it's zstd-compressed.
    """
    f = open(filename, mode)
    if filename.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(f)
    return f

def read_records(stream):
    while True:
        header_size = struct.calcsize('<5i')
        header = stream.read(struct.calcsize('<5i'))
        if len(header) < header_size:
            break

        scope_len, name_len, dtype_len, shape_len, arr_buffer_len = struct.unpack('<5i', header)

        block_base_fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
        block_base = stream.read(struct.calcsize(block_base_fmt))
        block_arr = stream.read(arr_buffer_len)

        scope, name, dtype, *shape = struct.unpack(block_base_fmt, block_base)
        scope, name, dtype = (a.decode('utf-8') for a in [scope, name, dtype])
        arr = np.frombuffer(block_arr, dtype=dtype)
        arr = np.reshape(arr, shape)
        if sys.byteorder == 'big':
            arr = arr.byteswap()

        yield scope, name, arr

def load_jax_params(filename):
    if not filename.endswith('.bin.zst') and not filename.endswith('.bin'):
        raise ValueError('Filename should either end in ".bin" or ".zst".')
    with open_jax_params(filename, 'rb') as f:
        params = {}
        for scope, name, arr in read_records(f):
            if scope not in params:
                params[scope] = {}

            if arr.dtype == 'bfloat16':
                arr = arr.astype(float)

            params[scope][name] = torch.tensor(arr)

    return params




def main():
    jax_weights = load_jax_params('/Users/kilianmandon/Projects/alphafold3/data/af3.bin.zst')
    pytorch_weights = remap_params(jax_weights)
    torch.save(pytorch_weights, 'data/params/af3_pytorch.pt')



if __name__ == '__main__':
    main()
