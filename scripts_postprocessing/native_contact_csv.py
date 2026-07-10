import argparse
import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def create_ag_parser():
    parser = argparse.ArgumentParser(description="Program to create plots of contact analysis evolution")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path of the trajectory to analyze")
    parser.add_argument("-to", "--topology", type=str, required=True, help="Path of the topology to analyze")
    parser.add_argument("-o", "--output", type=str, default="native_contact.csv", help="Output filename (with extension)")
    return parser


def best_hummer_q_protein_protein(traj, native, chainA=0, chainB=1):
    BETA_CONST = 50
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45

    heavy_A = native.topology.select(f'chainid {chainA} and not element H')
    heavy_B = native.topology.select(f'chainid {chainB} and not element H')

    native_pairs = np.array(list(product(heavy_A, heavy_B)))
    distances_native = md.compute_distances(native[0], native_pairs)[0]

    contact_mask = distances_native < NATIVE_CUTOFF
    native_contacts = native_pairs[contact_mask]
    print("Number of native inter-protein contacts:", len(native_contacts))

    if len(native_contacts) == 0:
        raise ValueError("No native inter-protein contacts found.")

    r = md.compute_distances(traj, native_contacts)
    r0 = md.compute_distances(native[0], native_contacts)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q

def main(trajectory, topology, output):
    print("Loading trajectory and native structure...")
    traj = md.load(trajectory, top=topology)
    native = md.load(topology)
    
    result = best_hummer_q_protein_protein(traj, native)

    time = traj.time
    data = {'Frame': time, 'Q value': result}
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    parser = create_ag_parser()
    args = parser.parse_args()
    main(args.trajectory, args.topology, args.output)
