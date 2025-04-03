from pathlib import Path

import torch

def main():
    all_diff_noise = []
    all_rand_rot = []
    all_rand_trans = []

    for i in range(29):
        diff_noise = torch.load(f'../alphafold3/kilian/pure_test_outputs/diff_noise_{i}.pt', weights_only=False)
        diff_rot = torch.load(f'../alphafold3/kilian/pure_test_outputs/diff_randaug_rot_{i}.pt', weights_only=False)
        diff_trans = torch.load(f'../alphafold3/kilian/pure_test_outputs/diff_randaug_trans_{i}.pt', weights_only=False)

        all_diff_noise.append(diff_noise)
        all_rand_rot.append(diff_rot)
        all_rand_trans.append(diff_trans)

    all_shuffle = [torch.load(f'../alphafold3/kilian/pure_test_outputs/rec_{i}_msa_shuffle_order.pt', weights_only=False) for i in range(11)]

    all_diff_noise = torch.stack(all_diff_noise, dim=0)
    all_rand_rot = torch.stack(all_rand_rot, dim=0)
    all_rand_trans = torch.stack(all_rand_trans, dim=0)
    all_shuffle = torch.stack(all_shuffle, dim=0)

    init_pos = torch.load(f'../alphafold3/kilian/pure_test_outputs/diff_initial_positions.pt', weights_only=False)

    torch.save(all_diff_noise, 'tests/test_lysozyme/debug_inputs/diffusion_noise.pt')
    torch.save(all_rand_rot, 'tests/test_lysozyme/debug_inputs/diffusion_randaug_rot.pt')
    torch.save(all_rand_trans, 'tests/test_lysozyme/debug_inputs/diffusion_randaug_trans.pt')
    torch.save(init_pos, 'tests/test_lysozyme/debug_inputs/diffusion_initial_pos.pt')
    torch.save(all_shuffle, 'tests/test_lysozyme/debug_inputs/msa_shuffle_order.pt')

if __name__=='__main__':
    main()
