#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from deeptime.decomposition import TICA


# ----------------------------
# CLI
# ----------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute tICA projections from feature descriptors (.dat/.csv/.npy/.npz)."
    )

    p.add_argument(
        "-i", "--inputs",
        required=True,
        nargs="+",
        help=("One or more feature files. "
              "If multiple files are given, they are treated as separate trajectories.")
    )

    p.add_argument("--delimiter", default=None,
                   help=("Delimiter for text files (.dat/.txt/.csv). "
                         "Default: auto (whitespace). Use ',' for CSV."))

    p.add_argument("--skiprows", type=int, default=0,
                   help="Skip this many header rows when reading text files (default: 0).")

    p.add_argument("--npz-key", default=None,
                   help=("If input is .npz, load this key. "
                         "If omitted, uses the first array in the archive."))

    p.add_argument("--lag", type=int, required=True,
                   help="tICA lag time (in frames).")

    p.add_argument("--dim", type=int, default=2,
                   help="Number of tIC dimensions to keep (default: 2). Use -1 for all.")

    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth frame from each input (default: 1).")

    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit to at most N frames per trajectory after stride (default: no limit).")

    p.add_argument("--drop-cols", type=int, nargs="*",
                   help=("Zero-based column indices to drop (e.g. time column). "
                         "Example: --drop-cols 0"))

    p.add_argument("--only-cols", type=int, nargs="*",
                   help=("Zero-based column indices to keep (overrides --drop-cols). "
                         "Example: --only-cols 2 3 4 5"))

    p.add_argument("--nan-policy", choices=["error", "drop-rows", "fill"], default="error",
                   help=("How to handle NaNs/inf in features: "
                         "'error' (default), 'drop-rows', or 'fill' (fill with column means)."))

    p.add_argument("--concatenate", action="store_true",
                   help=("Concatenate projected outputs from all trajectories into one array for saving. "
                         "Default: save one .npy per trajectory if multiple inputs."))

    p.add_argument("--out", default="tica.npy",
                   help=("Output .npy path. "
                         "If multiple inputs and not --concatenate, this becomes a prefix."))

    p.add_argument("--out-dat", default=None,
                   help=("Optional text output (.dat) path. "
                         "If multiple inputs and not --concatenate, this becomes a prefix."))

    p.add_argument("--dat-fmt", default="%.8f",
               help="Format for --out-dat (default: %%.8f).")


    p.add_argument("--tica-model-out", default=None,
                   help="Optional path to save the tICA model (pickle via pyemma). Example: tica_model.pkl")

    return p


# ----------------------------
# Loading helpers
# ----------------------------
def _as_2d(a: np.ndarray, where: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        # single feature column
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{where}: expected 2D array (n_frames, n_features). Got shape {a.shape}")
    return a


def load_features_one(path: Union[str, Path],
                      delimiter: Optional[str],
                      skiprows: int,
                      npz_key: Optional[str]) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _as_2d(arr, str(path))

    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if npz_key is not None:
            if npz_key not in z:
                raise KeyError(f"{path}: key '{npz_key}' not found. Available: {list(z.keys())}")
            arr = z[npz_key]
        else:
            keys = list(z.keys())
            if len(keys) == 0:
                raise ValueError(f"{path}: empty npz.")
            arr = z[keys[0]]
        return _as_2d(arr, str(path))

    # text-like
    # delimiter=None => whitespace
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    return _as_2d(arr, str(path))


def select_columns(x: np.ndarray,
                   only_cols: Optional[List[int]],
                   drop_cols: Optional[List[int]]) -> np.ndarray:
    if only_cols is not None and len(only_cols) > 0:
        return x[:, only_cols]
    if drop_cols is not None and len(drop_cols) > 0:
        mask = np.ones(x.shape[1], dtype=bool)
        for c in drop_cols:
            if c < 0 or c >= x.shape[1]:
                raise IndexError(f"drop col {c} out of range for n_features={x.shape[1]}")
            mask[c] = False
        return x[:, mask]
    return x


def handle_nans(x: np.ndarray, policy: str, where: str) -> np.ndarray:
    bad = ~np.isfinite(x)
    if not bad.any():
        return x

    if policy == "error":
        idx = np.argwhere(bad)
        # show a small hint
        r, c = idx[0].tolist()
        raise ValueError(f"{where}: found NaN/inf at row={r}, col={c}. Use --nan-policy drop-rows or fill.")

    if policy == "drop-rows":
        good_rows = np.isfinite(x).all(axis=1)
        x2 = x[good_rows]
        if x2.shape[0] == 0:
            raise ValueError(f"{where}: all rows removed due to NaN/inf.")
        return x2

    if policy == "fill":
        x2 = x.copy()
        # fill each column with mean of finite values
        for j in range(x2.shape[1]):
            col = x2[:, j]
            ok = np.isfinite(col)
            if not ok.any():
                raise ValueError(f"{where}: column {j} has no finite values to compute mean for fill.")
            m = col[ok].mean()
            col[~ok] = m
            x2[:, j] = col
        return x2

    raise ValueError(f"Unknown nan policy: {policy}")


def slice_traj(x: np.ndarray, stride: int, max_frames: Optional[int]) -> np.ndarray:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    x = x[::stride]
    if max_frames is not None:
        x = x[:max_frames]
    return x


# ----------------------------
# tICA
# ----------------------------
def run_tica(features: Union[np.ndarray, List[np.ndarray]], lag: int, dim: int):
    estimator = TICA(lagtime=lag, dim=dim)
    model = estimator.fit_fetch(features)
    if isinstance(features, list):
        outs = [model.transform(f) for f in features]
    else:
        outs = [model.transform(features)]
    return model, outs


# ----------------------------
# Saving helpers
# ----------------------------
def _is_multi(inputs: List[str]) -> bool:
    return len(inputs) > 1


def save_outputs(outputs: List[np.ndarray],
                 out: str,
                 out_dat: Optional[str],
                 dat_fmt: str,
                 concatenate: bool):
    if concatenate:
        cat = np.concatenate(outputs, axis=0)
        np.save(out, cat)
        if out_dat is not None:
            np.savetxt(out_dat, cat, fmt=dat_fmt)
        return

    # per-trajectory saving
    if len(outputs) == 1:
        np.save(out, outputs[0])
        if out_dat is not None:
            np.savetxt(out_dat, outputs[0], fmt=dat_fmt)
        return

    # multiple: treat out/out_dat as prefix
    out_path = Path(out)
    stem = out_path.stem
    suffix = out_path.suffix if out_path.suffix else ".npy"
    parent = out_path.parent if out_path.parent != Path("") else Path(".")

    for i, arr in enumerate(outputs):
        npy_name = parent / f"{stem}_traj{i}{suffix if suffix else '.npy'}"
        np.save(npy_name, arr)

    if out_dat is not None:
        outd = Path(out_dat)
        stemd = outd.stem
        parentd = outd.parent if outd.parent != Path("") else Path(".")
        for i, arr in enumerate(outputs):
            dat_name = parentd / f"{stemd}_traj{i}.dat"
            np.savetxt(dat_name, arr, fmt=dat_fmt)


def main(ca_distances, output, output_dat=None, lag=10, dim=2, stride=1,
         nan_policy="error", model_out=None, json_output=None):
    import json as _json
    import time

    print(f"Loading CA distances: {ca_distances}")
    x = load_features_one(ca_distances, delimiter=None, skiprows=0, npz_key=None)
    x = handle_nans(x, nan_policy, where=str(ca_distances))
    x = slice_traj(x, stride=stride, max_frames=None)

    if x.shape[0] < (lag + 1):
        raise ValueError(
            f"Not enough frames for lag={lag}. Need at least {lag + 1}, got {x.shape[0]}."
        )

    print(f"Running tICA: {x.shape[0]} frames, {x.shape[1]} features, lag={lag}, dim={dim}")
    t0 = time.time()
    tica_model, outs = run_tica([x], lag=lag, dim=dim)

    if model_out is not None:
        import pickle
        with open(model_out, "wb") as _f:
            pickle.dump(tica_model, _f)
        print(f"tICA model saved to {model_out}")

    save_outputs(outs, out=output, out_dat=output_dat, dat_fmt="%.8f", concatenate=False)
    print(f"tICA projection saved to {output}")

    result = {
        "step":         "tica",
        "status":       "completed",
        "output":       output,
        "n_frames":     int(outs[0].shape[0]),
        "n_components": int(outs[0].shape[1]),
        "lag":          lag,
        "dim":          dim,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    if output_dat:
        result["output_dat"] = output_dat
    if model_out:
        result["model"] = model_out

    if json_output:
        with open(json_output, "w") as f:
            _json.dump(result, f, indent=2)

    return result


def cli_main():
    args = make_parser().parse_args()

    trajs: List[np.ndarray] = []
    for f in args.inputs:
        x = load_features_one(f, delimiter=args.delimiter, skiprows=args.skiprows, npz_key=args.npz_key)
        x = select_columns(x, args.only_cols, args.drop_cols)
        x = handle_nans(x, args.nan_policy, where=str(f))
        x = slice_traj(x, stride=args.stride, max_frames=args.max_frames)

        if x.shape[0] < (args.lag + 1):
            raise ValueError(
                f"{f}: not enough frames after slicing for lag={args.lag}. "
                f"Need at least lag+1 frames, got {x.shape[0]}."
            )
        trajs.append(x)

    tica_model, outs = run_tica(trajs, lag=args.lag, dim=args.dim)

    if args.tica_model_out is not None:
        import pickle
        with open(args.tica_model_out, "wb") as _f:
            pickle.dump(tica_model, _f)

    save_outputs(outs, out=args.out, out_dat=args.out_dat, dat_fmt=args.dat_fmt, concatenate=args.concatenate)

    shapes = ", ".join([str(o.shape) for o in outs])
    print(f"Done. tICA outputs shapes: {shapes}")


if __name__ == "__main__":
    cli_main()
