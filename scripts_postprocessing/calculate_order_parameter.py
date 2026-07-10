#!/usr/bin/env python3
"""
Per-residue nematic order parameter (S2) from CA atoms.

For each residue i, S2 is computed from a sliding window of consecutive
CA-CA bond vectors centred at that residue using mdtraj.compute_nematic_order.
Each CA_j -> CA_{j+1} pair defines a local backbone direction; the nematic
order of those vectors within the window measures how locally aligned the
backbone is.

A global S2 per frame (all CA-CA pairs) is also written as a time-series.
"""
import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd


def create_ag_parser():
    parser = argparse.ArgumentParser(
        description="Calculate per-residue nematic order parameter (S2) from CA atoms"
    )
    parser.add_argument("-t", "--trajectory", type=str, required=True,
                        help="Path to the trajectory (.dcd)")
    parser.add_argument("-to", "--topology", type=str, required=True,
                        help="Path to the topology (.pdb/.cif)")
    parser.add_argument("-o", "--output", type=str, default="order_parameter.csv",
                        help="Output CSV for per-residue S2 (default: order_parameter.csv)")
    parser.add_argument("--global_out", type=str, default="order_parameter_global.dat",
                        help="Output .dat for global S2 per frame (default: order_parameter_global.dat)")
    parser.add_argument("--window", type=int, default=5,
                        help="Half-window size in residues for local S2 (default: 5)")
    parser.add_argument("--chain", type=int, default=0,
                        help="Chain index to analyse (default: 0)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to write JSON result file")
    return parser


def main(trajectory, topology, output="order_parameter.csv",
         global_out="order_parameter_global.dat", window=5,
         chain=0, json_output=None):

    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    # Select CA atoms for the requested chain
    ca_indices = traj.topology.select(f"name CA and chainid {chain}")
    if len(ca_indices) < 2:
        raise ValueError(f"Fewer than 2 CA atoms found for chain {chain}")
    print(f"Found {len(ca_indices)} CA atoms in chain {chain}")

    # ------------------------------------------------------------------ #
    # Build list of consecutive CA-CA pairs (each defines a bond vector)  #
    # ca_pairs[k] = (ca_indices[k], ca_indices[k+1])                      #
    # ------------------------------------------------------------------ #
    ca_pairs = [[int(ca_indices[k]), int(ca_indices[k + 1])]
                for k in range(len(ca_indices) - 1)]

    # ------------------------------------------------------------------ #
    # Global S2: nematic order of ALL CA-CA vectors per frame             #
    # ------------------------------------------------------------------ #
    print("Computing global S2 per frame...")
    s2_global = md.compute_nematic_order(traj, indices=ca_pairs)  # (n_frames,)
    np.savetxt(global_out, np.column_stack([np.arange(traj.n_frames), s2_global]),
               header="frame  S2_global", fmt=["%d", "%.6f"])
    print(f"Global S2 saved to {global_out}")

    # ------------------------------------------------------------------ #
    # Per-residue S2: sliding window of CA-CA pairs centred at residue i  #
    # Residue i has pairs from max(0, i-window) to min(n-2, i+window-1)  #
    # ------------------------------------------------------------------ #
    n_ca = len(ca_indices)
    records = []

    print(f"Computing per-residue S2 with window={window}...")
    for i, ca_idx in enumerate(ca_indices):
        atom = traj.topology.atom(ca_idx)

        # Window of pair indices (indices into ca_pairs list)
        pair_start = max(0, i - window)
        pair_end   = min(n_ca - 1, i + window)   # exclusive upper bound for pairs

        local_pairs = ca_pairs[pair_start:pair_end]

        if len(local_pairs) < 2:
            # Not enough vectors for a meaningful order parameter
            s2_mean = np.nan
            s2_std  = np.nan
        else:
            s2_local = md.compute_nematic_order(traj, indices=local_pairs)  # (n_frames,)
            s2_mean  = float(np.mean(s2_local))
            s2_std   = float(np.std(s2_local))

        records.append({
            "residue_index": atom.residue.index,
            "residue_name":  atom.residue.name,
            "residue_seq":   atom.residue.resSeq,
            "chain":         atom.residue.chain.index,
            "S2_mean":       s2_mean,
            "S2_std":        s2_std,
        })

    df = pd.DataFrame(records)
    df.to_csv(output, index=False)
    print(f"Per-residue S2 saved to {output}")

    result = {
        "step":              "order_parameter",
        "status":            "completed",
        "output":            output,
        "global_output":     global_out,
        "n_residues":        len(records),
        "n_frames":          int(traj.n_frames),
        "window":            window,
        "chain":             chain,
        "mean_S2_global":    float(np.mean(s2_global)),
        "mean_S2_per_res":   float(np.nanmean(df["S2_mean"])),
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(
        args.trajectory,
        args.topology,
        args.output,
        args.global_out,
        args.window,
        args.chain,
        args.json,
    )
