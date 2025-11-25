# src/chem_error_corr.py

import os
import yaml
from pathlib import Path

def _resolve_l_max() -> int:
    # 1. Dynamic override from run_all.py (highest priority)
    override = os.getenv('VQC_L_MAX_OVERRIDE')
    if override is not None:
        val = int(override)
        print(f"L_max ← VQC_L_MAX_OVERRIDE={val} (dynamic override)")
        return val

    # 2. CLI --L_max / --l_max
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

    # 3. YAML fallback
    yaml_path = Path(__file__).parent.parent / 'configs' / 'params.yaml'
    if yaml_path.exists():
        try:
            cfg = yaml.safe_load(yaml_path.read_text()) or {}
            val = cfg.get('qubit_multi', {}).get('L_max', 25)
            print(f"L_max ← configs/params.yaml → {val}")
            return int(val)
        except Exception as e:
            print(f"YAML load failed: {e}")

    print("L_max ← default = 25")
    return 25

L_max = _resolve_l_max()  # ← GLOBAL, resolved ONCE at import time
print(f"Final effective L_max = {L_max}\n")

import numpy as np
import pandas as pd
import pyscf
from typing import Dict, Any, Tuple
import psutil
import time
import argparse
import re

# GLOBAL WARNING SUPPRESSION
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# === QEC MODE DETECTION ===
qec_level = int(os.getenv('QEC_LEVEL', '16'))
qec_8qubit = os.getenv('VQC_QEC_8QUBIT', 'false').lower() == 'true'
qec_16qubit = os.getenv('VQC_QEC_16QUBIT', 'false').lower() == 'true'

effective_mode = f"{qec_level}-QUBIT" if qec_level >= 16 else "8-QUBIT"
print(f"▓▒░ {effective_mode} QEC ░▒▓")


# =============================================================================
# Main QEC Simulation Function
# =============================================================================
def run_chem_qec(params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    start_time = time.time()

    # NUMA Affinity - Bind to cores 0–35 (Socket 0 on Dual E5-2699 v3)
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(list(range(36)))
        print("Chem QEC: NUMA affinity set to cores 0–35 (Socket 0)")
    except Exception as e:
        print(f"Warning: Could not set CPU affinity ({e}); continuing without.")

    # Real PySCF: H2 ground state (RHF/sto-3g)
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    # Coupling strength with OAM-dependent detuning
    alpha_base = 0.03
    alpha_capped = alpha_base * (1 + min(L_max, 50) * 0.01)
    alpha = alpha_capped

    # Time grid
    time_ns = np.linspace(0, 100.0, 100)
    data = []

    # QEC regime selection
    if qec_16qubit:
        qec_factor = 1e-4
        alpha_base = 0.0015
        print("▓▒░ 16-QUBIT ENGAGED ░▒▓")
        print("QEC STATUS: 16-QUBIT DOMINANCE → factor=1e-4 | α→0.0015")
    elif qec_8qubit:
        qec_factor = 0.05
        alpha_base = 0.035
        print("QEC STATUS: 8-Qubit legacy fallback")
    else:
        qec_factor = 1.0
        alpha_base = 0.1

    # Asymptotic L-shielding
    alpha = alpha_base * (1.0 + 50.0 / (L_max + 100.0))

    # Main simulation loop
    for time_val in time_ns:
        chem_error = np.random.exponential(0.8) + 0.01 * np.sin(10 * time_val) + 0.05
        corrected_error = chem_error * qec_factor * np.exp(-alpha * time_val)
        fidelity = np.clip(1.0 - corrected_error * 0.1, 0.0, 1.0)

        data.append({
            'time_ns': time_val,
            'chem_error': chem_error,
            'corrected_error': corrected_error,
            'fidelity': fidelity
        })

    df = pd.DataFrame(data)
    runtime = time.time() - start_time
    mean_fid = df['fidelity'].mean()

    regime = "16-QUBIT QEC" if qec_16qubit else ("8-QUBIT QEC" if qec_8qubit else "NO QEC")

    print(f"Chem QEC complete | Regime: {regime} | "
          f"L_max={L_max} | α≈{alpha:.6f} | mean FID={mean_fid:.10f} | "
          f"runtime={runtime:.2f}s")

    summary = {
        "regime": regime,
        "L_max": L_max,
        "alpha": alpha,
        "mean_fid": mean_fid,
        "runtime": runtime
    }
    return df, summary


# =============================================================================
# Standalone CLI + Clean Output
# =============================================================================
if __name__ == "__main__":
    # Consume --L_max early to avoid conflicts
    parser = argparse.ArgumentParser(description="Standalone Chem QEC Simulator → CSV Export")
    parser.add_argument("--L_max", "--l_max", type=int, help=argparse.SUPPRESS)
    parser.parse_args()

    # Load params
    config_path = Path(__file__).parent.parent / "configs" / "params.yaml"
    if config_path.exists():
        with open(config_path) as f:
            params = yaml.safe_load(f) or {}
        print(f"Loaded full params from {config_path}")
    else:
        params = {'fidelity': {'qec_4qubit': True}}
        print(f"Warning: {config_path} not found → using minimal defaults")

    # Ensure output directory
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation – now returns everything needed
    df, summary = run_chem_qec(params)

    # Export CSV with clean naming (strips any old _L## tags)
    tentative_path = output_dir / f"chem_qec_L{L_max}.csv"
    cleaned_name = re.sub(r'_L\d+', '', tentative_path.stem)
    final_csv_path = output_dir / f"{cleaned_name}_L{L_max}.csv"
    df.to_csv(final_csv_path, index=False)

    # Final clean summary
    print(f"CSV exported → {final_csv_path}")
    print(f"   Shape: {df.shape} | "
          f"Regime: {summary['regime']} | "
          f"L_max={summary['L_max']} | "
          f"α≈{summary['alpha']:.6f} | "
          f"mean FID={summary['mean_fid']:.10f} | "
          f"runtime={summary['runtime']:.2f}s")

# eof