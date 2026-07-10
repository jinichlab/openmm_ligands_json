#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET

import mdtraj as md
import numpy as np


# ---- XML schema tags (your run.xml) ----
PHASE_TAGS = [
    "conventional-md-prep",
    "conventional-md",
    "gamd-equilibration-prep",
    "gamd-equilibration",
    "gamd-production",
]


def create_ag_parser():
    p = argparse.ArgumentParser(
        description="Compute Best-Hummer Q for a single protein and write a tutorial-style .dat CV. "
                    "Optionally restrict output to preparation or production frames using run XML."
    )
    p.add_argument("-t", "--trajectory", required=True, help="Trajectory (e.g., .dcd)")
    p.add_argument("-to", "--topology", required=True, help="Topology/native (e.g., .pdb)")
    p.add_argument("--q-out", default=None,
                   help="Optional output path for Best-Hummer Q (fraction of native contacts) .dat.")

    # Q definition parameters
    p.add_argument("--chainid", type=int, default=0, help="Chain index to analyze (default: 0)")
    p.add_argument("--native-cutoff", type=float, default=0.45, help="Native contact cutoff in nm (default: 0.45)")
    p.add_argument("--beta", type=float, default=50.0, help="Best-Hummer beta (default: 50)")
    p.add_argument("--lam", type=float, default=1.8, help="Best-Hummer lambda (default: 1.8)")
    p.add_argument("--min-seq-sep", type=int, default=3,
                   help="Minimum residue separation for a contact (default: 3)")

    # X-axis formatting
    p.add_argument("--x-axis", choices=["time", "frame"], default="time",
                   help="Write x-axis as 'time' (ps if available) or 'frame' (index).")

    # XML-based phase filtering
    p.add_argument("-x", "--xml", default=None,
                   help="Run XML containing <dt> and <number-of-steps> phases (optional).")
    p.add_argument("--phase", choices=["all", "prep", "prod"], default="all",
                   help="Which part to output (requires --xml for prep/prod). "
                        "'prep' = before GaMD production starts; 'prod' = production only.")

    # Optional trajectory time step if traj.time isn't populated and you want time axis in ps
    p.add_argument("--dt-ps-fallback", type=float, default=None,
                   help="If traj.time is missing/zeros and --x-axis time, use this dt (ps) "
                        "to compute time = frame_index * dt_ps.")

    # Options for calculation of CVs
    p.add_argument("--rmsd-out", default=None,
                   help="Optional output path for RMSD to native (dat). If set, writes rmsd (nm).")
    p.add_argument("--rmsd-sel", type=str, default=None,
                   help="MDTraj selection string for RMSD calculation "
                        "(default: Cα atoms of --chainid). "
                        "Example: 'chainid 0 and backbone'")

    p.add_argument("--ca-dist-out", default=None,
                   help="Optional output path for pairwise Cα distances (npy). If set, saves (n_frames, n_pairs) in nm.")

    p.add_argument("--ca-min-seq-sep", type=int, default=3,
                   help="Minimum residue separation for Cα distance pairs (default: 3).")

    p.add_argument("--phi-psi-out", default=None,
                   help="Optional output path for combined phi/psi angles (dat). Writes x, then phi columns, then psi columns.")

    p.add_argument("--angles-deg", action="store_true",
                   help="If set, write phi/psi in degrees instead of radians.")

    p.add_argument("--rg-out", default=None,
                   help="Optional output path for radius of gyration (dat). "
                        "Writes Rg in nm per frame. Use --rg-sel to choose atoms.")
    p.add_argument("--rg-sel", type=str, default=None,
                   help="MDTraj selection string for Rg calculation "
                        "(default: Cα atoms of --chainid). "
                        "Example: 'chainid 0 and not element H'")

    return p


def _find_text(root: ET.Element, tag: str) -> str | None:
    e = root.find(f".//{tag}")
    if e is None or e.text is None:
        return None
    return e.text.strip()


def _req_int(root: ET.Element, tag: str) -> int:
    txt = _find_text(root, tag)
    if txt is None:
        raise ValueError(f"Missing <{tag}> in XML.")
    try:
        v = int(float(txt))
    except ValueError:
        raise ValueError(f"Could not parse <{tag}>{txt}</{tag}> as int.")
    if v < 0:
        raise ValueError(f"<{tag}> must be nonnegative, got {v}.")
    return v


def _req_float(root: ET.Element, tag: str) -> float:
    txt = _find_text(root, tag)
    if txt is None:
        raise ValueError(f"Missing <{tag}> in XML.")
    try:
        v = float(txt)
    except ValueError:
        raise ValueError(f"Could not parse <{tag}>{txt}</{tag}> as float.")
    return v

def read_run_xml(xml_path: str) -> dict:
    """
    Returns:
      dt_ps: float (ps per integrator step)
      boundaries: list of dicts with start_step/end_step for each phase tag
      prod_start_step: cumulative start step of gamd-production
      prod_end_step: cumulative end step of gamd-production

      steps_per_frame: int | None
          Integrator steps between saved coordinate frames (trajectory stride) inferred from <outputs>.
      dt_frame_ps: float | None
          Picoseconds between saved coordinate frames (steps_per_frame * dt_ps).
      stride_source: str | None
          Which XML path was used to infer steps_per_frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dt_ps = _req_float(root, "dt")  # ps per integrator step
    if dt_ps <= 0:
        raise ValueError(f"<dt> must be positive, got {dt_ps}.")

    phase_steps = {tag: _req_int(root, tag) for tag in PHASE_TAGS}

    boundaries = []
    cur = 0
    for tag in PHASE_TAGS:
        start = cur
        end = cur + phase_steps[tag]
        boundaries.append({"tag": tag, "start_step": start, "end_step": end, "nsteps": phase_steps[tag]})
        cur = end

    prod_start_step = next(b["start_step"] for b in boundaries if b["tag"] == "gamd-production")
    prod_end_step = next(b["end_step"] for b in boundaries if b["tag"] == "gamd-production")

    # ---- NEW: infer coordinate stride (steps between saved frames) from <outputs> ----
    # Try coordinate interval first (if present), then fall back to energy/statistics.
    stride_candidates = [
        ("outputs/reporting/coordinates/interval", ".//outputs/reporting/coordinates/interval"),
        ("outputs/reporting/energy/interval", ".//outputs/reporting/energy/interval"),
        ("outputs/reporting/statistics/interval", ".//outputs/reporting/statistics/interval"),
    ]

    steps_per_frame = None
    stride_source = None
    for label, xpath in stride_candidates:
        e = root.find(xpath)
        if e is not None and e.text is not None and e.text.strip() != "":
            try:
                v = int(float(e.text.strip()))
            except ValueError:
                raise ValueError(f"Could not parse <{label}>{e.text}</{label}> as int.")
            if v <= 0:
                raise ValueError(f"<{label}> must be positive, got {v}.")
            steps_per_frame = v
            stride_source = label
            break

    dt_frame_ps = (steps_per_frame * dt_ps) if steps_per_frame is not None else None

    return {
        "dt_ps": dt_ps,
        "boundaries": boundaries,
        "prod_start_step": prod_start_step,
        "prod_end_step": prod_end_step,
        "steps_per_frame": steps_per_frame,
        "dt_frame_ps": dt_frame_ps,
        "stride_source": stride_source,
    }



def best_hummer_q_single_protein(
    traj: md.Trajectory,
    native: md.Trajectory,
    chainid: int = 0,
    beta: float = 50.0,
    lam: float = 1.8,
    native_cutoff: float = 0.45,
    min_seq_sep: int = 3,
) -> np.ndarray:
    """
    Best-Hummer Q for intra-protein native contacts (single chain), heavy atoms only.
    Native contacts are defined from native structure by cutoff and residue separation.
    """
    heavy = native.topology.select(f"chainid {chainid} and not element H")
    if heavy.size == 0:
        raise ValueError(f"No heavy atoms found for chainid={chainid}.")

    atom_to_res = np.array([native.topology.atom(i).residue.index for i in range(native.n_atoms)])

    pairs = []
    for i in range(len(heavy)):
        ai = heavy[i]
        ri = atom_to_res[ai]
        for j in range(i + 1, len(heavy)):
            aj = heavy[j]
            rj = atom_to_res[aj]
            if abs(ri - rj) >= min_seq_sep:
                pairs.append((ai, aj))

    if len(pairs) == 0:
        raise ValueError("No intra-chain atom pairs survived min_seq_sep filtering.")
    pairs = np.asarray(pairs, dtype=int)

    r0_all = md.compute_distances(native[0], pairs)[0]  # nm
    native_contacts = pairs[r0_all < native_cutoff]
    print("Number of native intra-protein contacts:", len(native_contacts))

    if len(native_contacts) == 0:
        raise ValueError(
            "No native intra-protein contacts found. Try increasing --native-cutoff or decreasing --min-seq-sep."
        )

    r = md.compute_distances(traj, native_contacts)           # (n_frames, n_contacts)
    r0 = md.compute_distances(native[0], native_contacts)[0]  # (n_contacts,)

    q = np.mean(1.0 / (1.0 + np.exp(beta * (r - lam * r0))), axis=1)
    return q

def compute_rmsd_to_native(
    traj: md.Trajectory, native: md.Trajectory, chainid: int = 0, sel: str = None
) -> np.ndarray:
    """
    RMSD (nm) of each frame to native[0].
    Default selection: Cα atoms of the chosen chain.
    Pass sel to override with any MDTraj selection string.
    """
    if sel is not None:
        atom_indices = traj.topology.select(sel)
        if atom_indices.size == 0:
            raise ValueError(f"RMSD selection '{sel}' matched no atoms.")
    else:
        atom_indices = traj.topology.select(f"chainid {chainid} and name CA")
        if atom_indices.size == 0:
            raise ValueError(f"No Cα atoms found for chainid={chainid}.")

    return md.rmsd(traj, native[0], atom_indices=atom_indices)


def compute_rg(traj: md.Trajectory, chainid: int = 0, sel: str = None) -> np.ndarray:
    """
    Radius of gyration (nm) for each frame.

    Parameters
    ----------
    traj    : md.Trajectory
    chainid : chain index used when sel is None (default: 0)
    sel     : MDTraj selection string. If None, uses Cα atoms of chainid.

    Returns
    -------
    rg : np.ndarray, shape (n_frames,), units nm
    """
    if sel is not None:
        atom_indices = traj.topology.select(sel)
        if atom_indices.size == 0:
            raise ValueError(f"Rg selection '{sel}' matched no atoms.")
    else:
        atom_indices = traj.topology.select(f"chainid {chainid} and name CA")
        if atom_indices.size == 0:
            raise ValueError(f"No Cα atoms found for chainid={chainid}.")

    return md.compute_rg(traj.atom_slice(atom_indices))


def compute_pairwise_ca_distances(
    traj: md.Trajectory, chainid: int = 0, min_seq_sep: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      distances_nm: (n_frames, n_pairs) float array
      pairs:        (n_pairs, 2) atom index pairs used (Cα atoms)
    """
    ca = traj.topology.select(f"chainid {chainid} and name CA")
    if ca.size == 0:
        raise ValueError(f"No Cα atoms found for chainid={chainid}.")

    # Map atom index -> residue index (topology residue indexing)
    atom_to_res = np.array([traj.topology.atom(i).residue.index for i in range(traj.n_atoms)])

    pairs = []
    for a_i_idx in range(len(ca)):
        ai = ca[a_i_idx]
        ri = atom_to_res[ai]
        for a_j_idx in range(a_i_idx + 1, len(ca)):
            aj = ca[a_j_idx]
            rj = atom_to_res[aj]
            if abs(ri - rj) >= min_seq_sep:
                pairs.append((ai, aj))

    if len(pairs) == 0:
        raise ValueError("No Cα pairs survived min_seq_sep filtering.")
    pairs = np.asarray(pairs, dtype=int)

    distances_nm = md.compute_distances(traj, pairs)  # (n_frames, n_pairs)
    return distances_nm, pairs

def compute_phi_psi_angles(traj: md.Trajectory, chainid: int = 0):
    """
    Compute backbone phi/psi for a given chain.

    Returns:
      phi_idx: (n_phi, 4) atom indices used by mdtraj
      phi:     (n_frames, n_phi) angles in radians
      psi_idx: (n_psi, 4) atom indices used by mdtraj
      psi:     (n_frames, n_psi) angles in radians
    """
    # md.compute_phi/psi returns (indices, angles)
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)

    # Restrict to the requested chain by checking the first atom of each dihedral
    # (all 4 atoms should be in the same residue/chain for standard phi/psi)
    if traj.topology.n_chains > 1:
        def _chain_of_atom(aidx: int) -> int:
            return traj.topology.atom(int(aidx)).residue.chain.index

        phi_keep = np.array([_chain_of_atom(d[0]) == chainid for d in phi_idx], dtype=bool)
        psi_keep = np.array([_chain_of_atom(d[0]) == chainid for d in psi_idx], dtype=bool)

        phi_idx, phi = phi_idx[phi_keep], phi[:, phi_keep]
        psi_idx, psi = psi_idx[psi_keep], psi[:, psi_keep]

    return phi_idx, phi, psi_idx, psi


def write_phi_psi_dat(path: str, phi: np.ndarray, psi: np.ndarray, unit: str):
    """
    Writes one file with: phi_0..phi_{nphi-1}, psi_0..psi_{npsi-1} (no time/frame column).
    phi shape: (n_frames, n_phi)
    psi shape: (n_frames, n_psi)
    """
    phi_cols = [f"phi({unit})_{i}" for i in range(phi.shape[1])]
    psi_cols = [f"psi({unit})_{i}" for i in range(psi.shape[1])]
    header = "# " + " ".join(phi_cols + psi_cols)

    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(phi.shape[0]):
            row_vals = np.concatenate([phi[i], psi[i]])
            row = " ".join([f"{v:12.6f}" for v in row_vals])
            f.write(row + "\n")


def write_cv_dat(path: str, y: np.ndarray):
    """Write only CV values (no time/frame column)."""
    with open(path, "w") as f:
        f.write("# Q\n")
        for yi in y:
            f.write(f"{yi:12.6f}\n")

def write_scalar_dat(path: str, y: np.ndarray, y_label: str):
    """Write only scalar CV values (no time/frame column)."""
    with open(path, "w") as f:
        f.write(f"# {y_label}\n")
        for yi in y:
            f.write(f"{yi:12.6f}\n")


def main(
    trajectory, topology,
    q_out=None, rmsd_out=None, ca_dist_out=None, phi_psi_out=None, rg_out=None,
    chainid=0, native_cutoff=0.45, beta=50.0, lam=1.8, min_seq_sep=3,
    rmsd_sel=None, ca_min_seq_sep=3, phi_psi_deg=False, rg_sel=None,
    json_output=None,
):
    """
    Compute all CVs in a single trajectory load. Each output is optional — pass a
    file path to enable it, or None to skip.

    Returns a result dict with the paths of all files written.
    """
    import json as _json

    requested = [q_out, rmsd_out, ca_dist_out, phi_psi_out, rg_out]
    if not any(requested):
        raise ValueError(
            "No CV output requested. Pass at least one of: "
            "q_out, rmsd_out, ca_dist_out, phi_psi_out, rg_out"
        )

    print("Loading trajectory and native structure...")
    traj = md.load(trajectory, top=topology)
    native = md.load(topology)
    print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    result = {
        "step": "cvs",
        "status": "completed",
        "n_frames": int(traj.n_frames),
        "outputs": {},
    }

    if q_out is not None:
        q = best_hummer_q_single_protein(
            traj, native, chainid=chainid, beta=beta, lam=lam,
            native_cutoff=native_cutoff, min_seq_sep=min_seq_sep,
        )
        write_cv_dat(q_out, q)
        result["outputs"]["q"] = q_out
        result["q_mean"] = float(q.mean())
        print(f"Saved Q to: {q_out}")

    if rmsd_out is not None:
        rmsd_nm = compute_rmsd_to_native(traj, native, chainid=chainid, sel=rmsd_sel)
        write_scalar_dat(rmsd_out, rmsd_nm, "RMSD(nm)")
        result["outputs"]["rmsd"] = rmsd_out
        result["rmsd_mean_nm"] = float(rmsd_nm.mean())
        print(f"Saved RMSD to: {rmsd_out}")

    if ca_dist_out is not None:
        ca_dists_nm, ca_pairs = compute_pairwise_ca_distances(
            traj, chainid=chainid, min_seq_sep=ca_min_seq_sep,
        )
        pairs_out = ca_dist_out.replace(".npy", "_pairs.npy")
        np.save(ca_dist_out, ca_dists_nm)
        np.save(pairs_out, ca_pairs)
        result["outputs"]["ca_distances"] = ca_dist_out
        result["outputs"]["ca_pairs"] = pairs_out
        result["n_ca_pairs"] = int(ca_pairs.shape[0])
        print(f"Saved Cα distances to: {ca_dist_out}  shape={ca_dists_nm.shape}")

    if phi_psi_out is not None:
        phi_idx, phi, psi_idx, psi = compute_phi_psi_angles(traj, chainid=chainid)
        if phi_psi_deg:
            phi = np.degrees(phi)
            psi = np.degrees(psi)
            unit = "deg"
        else:
            unit = "rad"
        write_phi_psi_dat(phi_psi_out, phi, psi, unit)
        result["outputs"]["phi_psi"] = phi_psi_out
        print(f"Saved phi/psi to: {phi_psi_out}")

    if rg_out is not None:
        rg_nm = compute_rg(traj, chainid=chainid, sel=rg_sel)
        write_scalar_dat(rg_out, rg_nm, "Rg(nm)")
        result["outputs"]["rg"] = rg_out
        result["rg_mean_nm"] = float(rg_nm.mean())
        print(f"Saved Rg to: {rg_out}")

    if json_output:
        with open(json_output, "w") as f:
            _json.dump(result, f, indent=2)

    return result


def _cli():
    """Original CLI entry point — unchanged behaviour."""
    args = create_ag_parser().parse_args()

    requested = [args.q_out, args.rmsd_out, args.ca_dist_out, args.phi_psi_out, args.rg_out]
    if not any(requested):
        create_ag_parser().error(
            "No CV output requested. Specify at least one of: "
            "--q-out, --rmsd-out, --ca-dist-out, --phi-psi-out, --rg-out"
        )

    print("Loading trajectory and native structure...")
    traj = md.load(args.trajectory, top=args.topology)
    native = md.load(args.topology)

    if args.phase != "all":
        if args.xml is None:
            raise ValueError("--phase prep/prod requires --xml.")
        xml_info = read_run_xml(args.xml)
        prod_start_step = xml_info["prod_start_step"]
        steps_per_frame_int = xml_info["steps_per_frame"]
        if steps_per_frame_int is None:
            raise ValueError("Could not infer steps_per_frame from XML.")
        prod_start_frame = int(np.ceil(prod_start_step / steps_per_frame_int))
        keep = (np.arange(traj.n_frames) < prod_start_frame) if args.phase == "prep" \
               else (np.arange(traj.n_frames) >= prod_start_frame)
        if not np.any(keep):
            raise ValueError("Phase filtering removed all frames.")
        traj = traj.slice(keep)
        print(f"[info] kept {traj.n_frames} frames for phase='{args.phase}'")

    if args.q_out is not None:
        q = best_hummer_q_single_protein(
            traj, native, chainid=args.chainid, beta=args.beta, lam=args.lam,
            native_cutoff=args.native_cutoff, min_seq_sep=args.min_seq_sep,
        )
        write_cv_dat(args.q_out, q)
        print(f"Saved Q to: {args.q_out}")

    if args.rmsd_out is not None:
        rmsd_nm = compute_rmsd_to_native(traj, native, chainid=args.chainid, sel=args.rmsd_sel)
        write_scalar_dat(args.rmsd_out, rmsd_nm, "RMSD(nm)")
        print(f"Saved RMSD to: {args.rmsd_out}")

    if args.ca_dist_out is not None:
        ca_dists_nm, ca_pairs = compute_pairwise_ca_distances(
            traj, chainid=args.chainid, min_seq_sep=args.ca_min_seq_sep,
        )
        np.save(args.ca_dist_out, ca_dists_nm)
        np.save(args.ca_dist_out.replace(".npy", "_pairs.npy"), ca_pairs)
        print(f"Saved Cα distances to: {args.ca_dist_out}")

    if args.phi_psi_out is not None:
        phi_idx, phi, psi_idx, psi = compute_phi_psi_angles(traj, chainid=args.chainid)
        if args.angles_deg:
            phi, psi = np.degrees(phi), np.degrees(psi)
            unit = "deg"
        else:
            unit = "rad"
        write_phi_psi_dat(args.phi_psi_out, phi, psi, unit)
        print(f"Saved phi/psi to: {args.phi_psi_out}")

    if args.rg_out is not None:
        rg_nm = compute_rg(traj, chainid=args.chainid, sel=args.rg_sel)
        write_scalar_dat(args.rg_out, rg_nm, "Rg(nm)")
        print(f"Saved Rg to: {args.rg_out}")


if __name__ == "__main__":
    _cli()
