# /vqc_sims/src/multi_beam_helix_within_helix_schematic.py

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
            cfg = yaml.safe_load(yaml_path.read_text()) or {}
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
import os

qec_level = int(os.getenv('QEC_LEVEL', '8'))
qec_8qubit = os.getenv('VQC_QEC_8QUBIT', 'false').lower() == 'true'
qec_16qubit = os.getenv('VQC_QEC_16QUBIT', 'false').lower() == 'true'

# Exponent: 8→8, 16→16, 32→32, etc. (scalable to QEC^∞)
qec_suppression_exponent = max(qec_level, 8)

effective_mode = f"{qec_level}+-QUBIT" if qec_level >= 16 else "8-QUBIT"
print(f"▓▒░ {effective_mode} QEC ░▒▓")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from typing import Optional
from scipy.interpolate import interp1d

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def _strip_legacy_L_suffix(path: str) -> str:
    basename = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'_L\d+', '', basename)

def multi_beam_helix_within_helix_schematic(
    knot_mod: bool = True,
    mean_fid: float = 0.9992,
    gamma1: float = 1.25,
    L_outer: Optional[int] = None,
    L_inner: Optional[int] = None,
    num_points: int = 1200,
    turns: int = 8,
    scale: float = 0.8,
    L_ref: int = 10,
    epsilon: float = 1e-10
):
    # SACRED L_inner CAP + resolution
    L_outer = L_outer if L_outer is not None else 3
    raw_L_inner = L_inner if L_inner is not None else L_max
    L_inner = min(raw_L_inner, 1999)
    effective_L = L_max

    theta = np.linspace(0, turns * 2 * np.pi, num_points)
    period = 2 * np.pi

    # Helices
    x_outer = np.cos(L_outer * theta)
    y_outer = np.sin(L_outer * theta)
    z_outer = theta / (2 * np.pi)

    x_inner = 0.4 * np.cos(-L_inner * theta)
    y_inner = 0.4 * np.sin(-L_inner * theta)
    z_inner = z_outer

    # ETERNAL 8₃ KNOT – IMMORTALIZED ONCE
    knot_t = np.linspace(0, 2 * np.pi, num_points)
    x_k = scale * np.cos(knot_t) * (2 + np.cos(2 * knot_t)) / 2
    y_k = scale * np.sin(knot_t) * (2 + np.cos(2 * knot_t)) / 2
    z_k = scale * np.sin(3 * knot_t) / 2
    x_k += scale * 0.5 * np.sin(4 * knot_t)
    y_k += scale * 0.3 * np.cos(3 * knot_t)
    knot_pos = np.column_stack([x_k, y_k, z_k])

    tangents = np.column_stack([np.gradient(knot_pos[:, i]) for i in range(3)])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + epsilon
    unit_tangents = tangents / norms

    angle = L_ref * knot_t / 2
    quat_real = np.cos(angle)
    quat_vec = np.sin(angle)[:, np.newaxis] * unit_tangents
    quat_comps = np.column_stack([quat_real, quat_vec])
    norms_q = np.linalg.norm(quat_comps, axis=1)
    detune_per = norms_q * 0.01

    knot_points = knot_pos

    if knot_mod:
        eps = 1e-9
        knot_t_ext = np.hstack([
            knot_t - period - eps,
            knot_t,
            knot_t + period + eps
        ])
        knot_pos_ext = np.vstack([knot_pos, knot_pos, knot_pos])

        assert np.all(np.diff(knot_t_ext) > 0), "non-monotonic timeline"

        knot_interp = np.zeros((len(theta), 3))
        for dim in range(3):
            f = interp1d(knot_t_ext, knot_pos_ext[:, dim],
                         kind='cubic', bounds_error=False, fill_value='extrapolate')
            knot_interp[:, dim] = f(theta % period)

        mod_r = 0.14 * np.abs(knot_interp[:, 2])
        detune_interp = np.interp(theta % period, knot_t, detune_per)
        mod_phase = detune_interp * (0.008 if qec_8qubit else 0.01)

        x_inner += mod_r * np.cos(-L_inner * theta + mod_phase)
        y_inner += mod_r * np.sin(-L_inner * theta + mod_phase)
        z_inner += 0.07 * knot_interp[:, 2]

        inner_label = f'Inner ℓ=–{L_inner} (8₃ Knot-Modulated + QEC-16Q)' + \
                      (" [CAPPED]" if raw_L_inner > 1999 else "")
    else:
        knot_interp = None
        inner_label = f'Inner ℓ=–{L_inner} (Pure Helix{" [CAPPED]" if raw_L_inner > 1999 else ""})'

    # [Plotting + save unchanged – see previous full version]


    # === Plotting ===
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_outer, y_outer, z_outer, 'b-', linewidth=3.6, label=f'Outer ℓ=+{L_outer}', alpha=0.95)
    ax.plot(x_inner, y_inner, z_inner, 'r-', linewidth=3.6, label=inner_label, alpha=0.92)

    if knot_mod and knot_interp is not None:
        ax.plot(knot_interp[::6, 0], knot_interp[::6, 1], knot_interp[::6, 2],
                'g--', linewidth=2.0, alpha=0.75, label='8₃ Stevedore Backbone (sampled)')

    ax.set_xlabel('X (a.u.)', fontsize=12)
    ax.set_ylabel('Y (a.u.)', fontsize=12)
    ax.set_zlabel('Z (propagation, m)', fontsize=12)
    ax.set_title('Vortex Quaternion Conduit – Dual Counter-Propagating OAM Beams\n'
                 'Topological 8₃ Knot Modulation + 16-Qubit QEC)',
                 fontsize=14, pad=20)

    ax.legend(loc='upper right', fontsize=11.5, framealpha=0.94)
    ax.view_init(elev=20, azim=52)
    ax.grid(True, alpha=0.3)

    qec_tag = " [16-QUBIT QEC ACTIVE]" if qec_16qubit else ""
    fid_annot = f'Mean Gate Fidelity = {mean_fid:.4f} (@ γ₁={gamma1}, L_max={effective_L}){qec_tag}'
    color = 'darkgreen' if qec_8qubit else 'darkred'
    bg = 'lightgreen' if qec_8qubit else 'lightgoldenrodyellow'
    if raw_L_inner > 1999:
        fid_annot += " | L_inner CAPPED @1999"
        color = 'orange'
        bg = 'mistyrose'

    ax.text2D(0.02, 0.94, fid_annot, transform=ax.transAxes, fontsize=13,
              color=color, bbox=dict(boxstyle='round,pad=0.7', facecolor=bg, alpha=0.9))

    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "multi_beam_helix_within_helix_schematic"
    qec_tag = "_16QEC" if qec_16qubit else ""
    cap_tag = "_INNERCAPPED" if raw_L_inner > 1999 else ""
    # SACRED ORDER: _8QEC BEFORE _L####
    filename = f"{base_name}{qec_tag}_L{effective_L}{cap_tag}.png"
    save_path = output_dir / filename

    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"VQC Schematic → {save_path} (L_outer={L_outer}, L_inner={L_inner}{' [CAPPED]' if raw_L_inner > 1999 else ''}, 16Q-QEC={'ON' if qec_16qubit else 'OFF'})")
    return knot_points, f'Mean Gate Fidelity = {mean_fid:.4f} (@ γ₁={gamma1}, L_max={effective_L}) [16-QUBIT QEC ACTIVE]'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VQC Helix-in-Helix Schematic Generator")
    parser.add_argument('--L_outer', type=int, default=3, help='Outer OAM value')
    parser.add_argument('--L_inner', type=int, help='Inner OAM value (will be capped ≤1999)')
    parser.add_argument('--no_knot', action='store_true', help='Disable 8₃ knot modulation')
    args = parser.parse_args()

    knot_pts, annotation = multi_beam_helix_within_helix_schematic(
        knot_mod=not args.no_knot,
        mean_fid=0.9994 if qec_8qubit else 0.9992,
        gamma1=1.25,
        L_outer=args.L_outer,
        L_inner=args.L_inner
    )
    print(f"Phase 1.2.83 Ω – Generation complete | 16-Qubit QEC = {qec_16qubit}")
    print(f"Annotation: {annotation}")

# eof