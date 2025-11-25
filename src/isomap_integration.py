# /vqc_sims/src/isomap_integration.py

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
        except:
            pass

    print("L_max ← default = 25")
    return 25

L_max = _resolve_l_max()  # ← GLOBAL, resolved ONCE at import
print(f"Final effective L_max = {L_max}\n")
# ============================================================

# === 16-QUBIT ===
import os

qec_level = int(os.getenv('QEC_LEVEL', '8'))
qec_8qubit = os.getenv('VQC_QEC_8QUBIT', 'false').lower() == 'true'
qec_16qubit = os.getenv('VQC_QEC_16QUBIT', 'false').lower() == 'true'

# Exponent: 8→8, 16→16, 32→32, etc. (scalable to QEC^∞)
qec_suppression_exponent = max(qec_level, 8)

effective_mode = f"{qec_level}-QUBIT" if qec_level >= 16 else "8-QUBIT"
print(f"▓▒░ {effective_mode} QEC ░▒▓")

import re
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import SparseEfficiencyWarning
from glob import glob
from typing import List, Tuple, Dict, Any, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

# GLOBAL WARNING SUPPRESSION – add to TOP of every src/*.py & analysis/*.py
import warnings
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)  # ← ETERNAL SILENCE ENFORCED
warnings.filterwarnings('ignore', category=RuntimeWarning)


def drop_zero_var_feats(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """Drop zero-variance features."""
    if data.shape[1] == 0:
        return data, 0
    selector = VarianceThreshold(threshold=1e-12)
    selector.fit(data)
    kept = selector.get_support()
    n_dropped = data.shape[1] - np.sum(kept)
    return data[:, kept], n_dropped


def drop_high_corr_feats(data: np.ndarray, corr_thresh: float = 0.98) -> Tuple[np.ndarray, int]:
    """Drop highly correlated features (> corr_thresh)."""
    if data.shape[0] < 2 or data.shape[1] <= 1:
        return data, 0
    corr_matrix = np.abs(np.corrcoef(data.T))
    np.fill_diagonal(corr_matrix, 0)
    to_drop = set()
    for i in range(corr_matrix.shape[0]):
        if i not in to_drop:
            high_corr = np.where(corr_matrix[i] > corr_thresh)[0]
            to_drop.update(high_corr)
    n_dropped = len(to_drop)
    print(f"High corr drop: {n_dropped} feats (thresh={corr_thresh:.3f})")
    keep_idx = [i for i in range(data.shape[1]) if i not in to_drop]
    return data[:, keep_idx], n_dropped


def compute_manual_stress(orig_dist: np.ndarray, emb: np.ndarray) -> float:
    """
    Compute raw stress using the true (geodesic) distance matrix from Isomap.
    Critical: orig_dist must be the full square distance matrix (not condensed).
    """
    if orig_dist is None or emb.shape[0] < 2:
        return np.nan
    emb_dist = squareform(pdist(emb))
    denom = np.sum(orig_dist**2) + 1e-12
    return np.sqrt(np.sum((orig_dist - emb_dist)**2) / denom)


def clean_basename(csv_path: str) -> str:
    """Nuclear strip of any legacy _L## before re-tagging."""
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    return re.sub(r'_L\d+', '', basename)


def apply_isomap_to_data(
    csv_path: str,
    n_components: int = 3,
    n_neighbors: Optional[int] = None,
    output_dir: str = 'outputs',
    min_samples: int = 2
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Core Isomap + preprocessing + nuclear-safe tagging + correct stress using geodesic distances."""
    try:
        df = pd.read_csv(csv_path)
        data = df.select_dtypes(include=[np.number]).values.astype(float)
        data = np.nan_to_num(data, nan=0.0)

        n_samples, n_feats = data.shape
        metrics = {
            'csv': os.path.basename(csv_path),
            'n_samples': n_samples,
            'n_features_raw': n_feats,
            'stress': np.nan
        }

        if n_samples < min_samples:
            print(f"Skip {csv_path}: <{min_samples} samples")
            return None, metrics

        # Preprocessing
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        data, n_zero = drop_zero_var_feats(data)
        metrics['zero_var_dropped'] = n_zero

        data, n_corr = drop_high_corr_feats(data)
        metrics['high_corr_dropped'] = n_corr

        # Isomap
        neighbors = n_neighbors or max(5, min(12, n_samples // 3))
        embedded = None
        dist_orig = None
        stress = np.nan

        try:
            iso = Isomap(n_components=n_components, n_neighbors=neighbors)
            embedded = iso.fit_transform(data)

            # Prefer scikit-learn's built-in stress_ if valid
            if hasattr(iso, 'stress_') and not np.isnan(getattr(iso, 'stress_', np.nan)):
                stress = float(iso.stress_)
            else:
                # Fallback: use the true geodesic distance matrix stored in Isomap
                dist_orig = getattr(iso, 'dist_matrix_', None)
                if dist_orig is not None and dist_orig.shape == (n_samples, n_samples):
                    stress = compute_manual_stress(dist_orig, embedded)
                else:
                    # Absolute worst case: fall back to Euclidean
                    stress = compute_manual_stress(squareform(pdist(data)), embedded)

        except Exception as e:
            print(f"Isomap failed ({e}), falling back to PCA...")
            pca = PCA(n_components=min(n_components, data.shape[1]))
            embedded = pca.fit_transform(data)
            stress = compute_manual_stress(squareform(pdist(data)), embedded)

        metrics['stress'] = float(stress)
        print(f"Isomap {os.path.basename(csv_path)} → stress={stress:.5f} (samples={n_samples})")

        # Plot + nuclear-safe save
        try:
            clean_name = clean_basename(csv_path)
            plot_path = os.path.join(output_dir, 'figures', f'isomap_embedding_{clean_name}_L{L_max}.png')
            os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

            if embedded.shape[1] >= 3:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                                c=np.arange(n_samples), cmap='viridis', s=40)
                fig.colorbar(sc, ax=ax, label='Sample Index')
                title = f'Isomap 3D • {clean_name}_L{L_max}'
                if qec_8qubit:
                    title += f' [{qec_level}-QUBIT QEC ACTIVE]'
                ax.set_title(f'{title}\nstress = {stress:.4f} | L_max={L_max}')
                ax.set_xlabel('Comp 1'); ax.set_ylabel('Comp 2'); ax.set_zlabel('Comp 3')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.figure(figsize=(10, 8))
                plt.scatter(embedded[:, 0], embedded[:, 1], c=np.arange(n_samples), cmap='viridis', s=40)
                plt.colorbar(label='Sample Index')
                title = f'Isomap 2D • {clean_name}_L{L_max}'
                if qec_8qubit:
                    title += f' [{qec_level}-QUBIT QEC ACTIVE]'
                plt.title(f'{title}\nstress = {stress:.4f} | L_max={L_max}')
                plt.xlabel('Component 1'); plt.ylabel('Component 2')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

            print(f"Plot saved → {plot_path}")
        except Exception as e:
            print(f"Plot save failed: {e}")

        return embedded, metrics

    except Exception as e:
        print(f"Isomap failed on {csv_path}: {e}")
        return None, {'stress': np.nan, 'csv': os.path.basename(csv_path)}


def batch_apply_isomap(
    paths: Optional[List[str]] = None,
    n_jobs: int = 1,
    n_components: int = 3,
    n_neighbors: Optional[int] = None,
    output_dir: str = 'outputs',
    metrics_csv: Optional[str] = None,
    min_samples: int = 2
) -> Tuple[List[Optional[np.ndarray]], Dict[str, Any]]:
    """Batch process with universal L_max and nuclear filename cleaning."""
    if paths is None:
        pattern = os.path.join(output_dir, 'tables', '*_L*.csv')
        all_csvs = glob(pattern)
        def extract_l(f):
            m = re.search(r'_L(\d+)', os.path.basename(f))
            return int(m.group(1)) if m else 0
        paths_to_process = sorted(
            [p for p in all_csvs if extract_l(p) <= L_max],
            key=extract_l, reverse=True
        )
        print(f"Auto-found {len(paths_to_process)} CSVs with L ≤ {L_max}")
    else:
        paths_to_process = [p for p in paths if isinstance(p, str) and os.path.exists(p)]
        print(f"Processing {len(paths_to_process)} provided paths")

    if not paths_to_process:
        print("No files to process")
        return [], {'mean_stress': np.nan, 'L_max_used': L_max}

    def process(p):
        return apply_isomap_to_data(
            p, n_components=n_components, n_neighbors=n_neighbors,
            output_dir=output_dir, min_samples=min_samples
        )

    print(f"Launching parallel Isomap (n_jobs={n_jobs}) with L_max={L_max}...")
    if qec_8qubit:
        print(f"░▒▓ {qec_level}-QUBIT QEC ▓▒░")

    results = Parallel(n_jobs=n_jobs)(delayed(process)(p) for p in paths_to_process)

    embeddings = []
    all_metrics = []
    for emb, met in results:
        if emb is not None:
            embeddings.append(emb)
        if isinstance(met, dict):
            all_metrics.append(met)

    agg = {
        'mean_stress': float(np.nanmean([m.get('stress', np.nan) for m in all_metrics])),
        'mean_n_samples': float(np.mean([m.get('n_samples', 0) for m in all_metrics])),
        'total_files': len(paths_to_process),
        'successful': len(all_metrics),
        'L_max_used': L_max,
        'qec_8qubit_active': qec_8qubit
    }

    if metrics_csv and all_metrics:
        try:
            df_log = pd.DataFrame([agg])
            mode = 'a' if os.path.exists(metrics_csv) else 'w'
            df_log.to_csv(metrics_csv, mode=mode, header=(mode=='w'), index=False)
            print(f"Batch metrics appended → {metrics_csv} | mean_stress={agg['mean_stress']:.4f}")
        except Exception as e:
            print(f"Metrics write failed: {e}")

    return embeddings, agg


if __name__ == "__main__":
    metrics_path = 'outputs/tables/vqc_metrics_batch.csv'
    embeddings, summary = batch_apply_isomap(
        paths=None,
        n_jobs=-1,  # use all cores
        n_components=3,
        n_neighbors=None,
        metrics_csv=metrics_path
    )
    qec_status = f" [{qec_level}-QUBIT QEC ACTIVE]" if qec_8qubit else ""
    print(f"\nBatch complete | L_max={L_max}{qec_status} | Valid embeddings: {len(embeddings)} | "
          f"Mean stress: {summary.get('mean_stress', np.nan):.4f}")

# eof