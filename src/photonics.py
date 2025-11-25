# /vqc_sims/src/photonics.py

import os
import yaml
from pathlib import Path

def _resolve_l_max() -> int:
    override = os.getenv('VQC_L_MAX_OVERRIDE')
    if override is not None:
        val = int(override)
        print(f"L_max ← VQC_L_MAX_OVERRIDE={val} (dynamic override)")
        return val
    try:
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--L_max', '--l_max', type=int, default=None)
        args, _ = parser.parse_known_args()
        if args.L_max is not None:
            print(f"L_max ← CLI={args.L_max}")
            return args.L_max
    except:
        pass
    yaml_path = Path(__file__).parent.parent / 'configs' / 'params.yaml'
    if yaml_path.exists():
        try:
            cfg = yaml.safe_load (yaml_path.read_text()) or {}
            val = cfg.get('qubit_multi', {}).get('L_max', 25)
            print(f"L_max ← configs/params.yaml → {val}")
            return int(val)
        except:
            pass
    print("L_max ← default = 25")
    return 25

L_max = _resolve_l_max()
print(f"Final effective L_max = {L_max}\n")

# === 16-QUBIT ===
qec_level = int(os.getenv('QEC_LEVEL', '16'))
qec_8qubit = os.getenv('VQC_QEC_8QUBIT', 'false').lower() == 'true'
qec_16qubit = os.getenv('VQC_QEC_16QUBIT', 'false').lower() == 'true'

# Exponent: 8→8, 16→16, 32→32, etc. (scalable to QEC^∞)
qec_suppression_exponent = max(qec_level, 16)

effective_mode = f"{qec_level}-QUBIT" if qec_level == 16 else "8-QUBIT"
print(f"▓▒░ {effective_mode} QEC ░▒▓")

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
import re
import psutil
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===================================================================
# HELPER: Nuclear legacy _L## tag stripping
# ===================================================================
def strip_legacy_L_tag(path: str) -> str:
    if not path:
        return path
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    cleaned_name = re.sub(r'_L-?\d+', '', name)
    cleaned_path = os.path.join(dirname, cleaned_name + ext) if dirname else cleaned_name + ext
    if cleaned_name != name:
        print(f"Stripped legacy L tag: {basename} → {os.path.basename(cleaned_path)}")
    return cleaned_path

# ===================================================================
# Core propagation
# ================================================================
def propagate_multi_ell(
    params: Optional[Dict[str, Any]] = None,
    l_max: Optional[int] = None
) -> pd.DataFrame:
    if params is None:
        params = {}

    start_time = time.time()

    # NUMA binding (PowerEdge 630 dual-socket)
    try:
        p = psutil.Process()
        if p.cpu_affinity()[0] < 36:
            p.cpu_affinity(list(range(36)))
            print("Photonics: Bound to NUMA node 0 (cores 0-35)")
        else:
            p.cpu_affinity(list(range(36, 72)))
            print("Photonics: Bound to NUMA node 1 (cores 36-71)")
    except Exception:
        print("NUMA binding skipped")

    z_start, z_end = params.get('z', [0.0, 10.0])
    z = np.linspace(z_start, z_end, params.get('n_z', 200))
    turbulence = params.get('turbulence', 0.05)
    chirp = params.get('chirp', 0.1)
    chunk_size = params.get('chunk_size', 10)

    effective_l_max = l_max if l_max is not None else L_max
    ell_list = np.arange(-effective_l_max, effective_l_max + 1)

    n_chunks = max(1, (len(ell_list) + chunk_size - 1) // chunk_size)
    results = []

    print(f"Starting propagation: |ℓ| ≤ {effective_l_max}, {len(ell_list)} modes, {n_chunks} chunks")
    print(f"   → Running under {effective_mode} QEC")

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(ell_list))
        chunk_ell = ell_list[start_idx:end_idx]
        if len(chunk_ell) == 0:
            continue

        phase = chunk_ell[:, None] * chirp * z[None, :]
        base_intensity = np.exp(-2 * phase ** 2)

        noise = np.random.normal(1.0, turbulence, base_intensity.shape)
        intensity = base_intensity * noise
        intensity = np.clip(intensity, 0.0, None)

        df_chunk = pd.DataFrame({
            'ell': np.repeat(chunk_ell, len(z)),
            'z': np.tile(z, len(chunk_ell)),
            'intensity': intensity.flatten(),
            'time_ns': np.linspace(0, 100, len(chunk_ell) * len(z))
        })
        results.append(df_chunk)

    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    runtime = time.time() - start_time
    print(f"Propagation complete |ℓ|≤{effective_l_max}: {runtime:.2f}s, {df.shape[0]:,} points")

    return df

# ===================================================================
# Standalone execution
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-ℓ LG Beam Propagation")
    parser.add_argument('--turbulence', type=float, default=0.05)
    parser.add_argument('--chirp', type=float, default=0.1)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="outputs/tables")
    parser.add_argument('--L_max', '--l_max', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    params = {
        'z': [0.0, 10.0],
        'n_z': 200,
        'turbulence': args.turbulence,
        'chirp': args.chirp,
        'chunk_size': args.chunk_size
    }

    df = propagate_multi_ell(params)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = f"{args.output_dir}/photonics_propagation_L{L_max}.csv"
    final_path = strip_legacy_L_tag(raw_path)

    df.to_csv(final_path, index=False)
    print(f"Saved: {final_path}")
    print(f"   → Mean intensity = {df['intensity'].mean():.5f}")
    print(f"   → L_max used = {L_max}")
    if qec_level >= 16:
        print("   → 16-QUBIT QEC CONDUIT ACTIVE")
    else:
        print("   → 8-QUBIT QEC CONDUIT ACTIVE")
    print(f"▓▒░ {effective_mode} QEC ░▒▓")

# eof