#!/usr/bin/env python3
import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Calculate DSSP secondary structure assignment per frame")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to the trajectory (.dcd)")
    parser.add_argument("-to", "--topology", type=str, required=True, help="Path to the topology (.pdb/.cif)")
    parser.add_argument("-o", "--output", type=str, default="dssp.csv",
                        help="Output CSV filename (default: dssp.csv)")
    parser.add_argument("--simplified", action="store_true",
                        help="Use simplified 3-state DSSP (H/E/C) instead of full 8-state")
    parser.add_argument("--summary", type=str, default=None,
                        help="Optional path for per-residue summary CSV (fraction of frames in each state)")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")
    return parser


def main(trajectory, topology, output="dssp.csv", simplified=False,
         summary_out=None, json_output=None):
    print(f"Loading trajectory: {trajectory}")
    traj = md.load(trajectory, top=topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    # MDTraj DSSP: returns array (n_frames, n_residues) with single-char codes
    # simplified=True → H, E, C; simplified=False → full DSSP 8-state
    dssp = md.compute_dssp(traj, simplified=simplified)
    print(f"DSSP computed: {traj.n_frames} frames × {dssp.shape[1]} residues")

    # Build per-frame CSV
    # Columns: time(ps), res_0, res_1, ..., res_N
    residues = [r for r in traj.topology.residues]
    col_names = ["time_ps"] + [f"{r.name}{r.resSeq}_{r.chain.index}" for r in residues]
    time_col = traj.time.reshape(-1, 1)
    data = np.concatenate([time_col, dssp], axis=1)
    df = pd.DataFrame(data, columns=col_names)
    df.to_csv(output, index=False)
    print(f"DSSP per-frame saved to {output}")

    # Optional per-residue summary
    if summary_out is not None:
        if simplified:
            states = ["H", "E", "C"]
        else:
            states = ["H", "B", "E", "G", "I", "T", "S", " "]

        summary_records = []
        for j, res in enumerate(residues):
            col = dssp[:, j]
            rec = {
                "residue_index": res.index,
                "residue_name": res.name,
                "residue_seq": res.resSeq,
                "chain": res.chain.index,
            }
            for s in states:
                rec[f"frac_{s}"] = float((col == s).mean())
            # dominant state
            rec["dominant_state"] = max(states, key=lambda s: (col == s).mean())
            summary_records.append(rec)

        df_sum = pd.DataFrame(summary_records)
        df_sum.to_csv(summary_out, index=False)
        print(f"DSSP per-residue summary saved to {summary_out}")

    result = {
        "step": "dssp",
        "status": "completed",
        "output": output,
        "summary": summary_out,
        "n_frames": int(traj.n_frames),
        "n_residues": int(dssp.shape[1]),
        "simplified": simplified,
    }

    if json_output:
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output, args.simplified,
         args.summary, args.json)
