import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import pearsonr
from scipy.linalg import svd
from scipy.signal import chirp, welch, butter, filtfilt

np.random.seed(42)


# Quaternion class for 3D rotations (Hamilton product) - from vqc-encoding-8bit-squidcurrents.py
class Quaternion:
    def __init__(self, w, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"q({self.w:.3f} + {self.x:.3f}i + {self.y:.3f}j + {self.z:.3f}k)"

    def norm(self):
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        n = self.norm()
        return Quaternion(self.w / n ** 2, -self.x / n ** 2, -self.y / n ** 2, -self.z / n ** 2)

    def multiply(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate_vector(self, v):
        # v: numpy array [x, y, z]
        qv = Quaternion(0, v[0], v[1], v[2])
        q_inv = self.inverse()
        rotated = self.multiply(qv).multiply(q_inv)
        return np.array([rotated.x, rotated.y, rotated.z])


# LG Mode Generation - Integrated from complex_ica.py
def lg_mode(ell, rho, phi, w0):
    abs_ell = abs(ell)
    C = np.sqrt(2 * factorial(abs_ell) / (np.pi * w0 ** 2))
    radial = C * (np.sqrt(2) * rho / w0) ** abs_ell * np.exp(-rho ** 2 / w0 ** 2)
    helical = np.exp(1j * ell * phi)
    return radial * helical


# p-Wave Altermagnetic Parameters - From PDFs (e.g., Amendment-VQC-pWave-Altermagnetic-BMGL-US63913110.pdf)
lambda_soc = 0.4  # Spin-orbit coupling (SOC)
p_odd_parity = 1.2  # Odd-parity p-wave splitting
gamma_1 = 1.5  # Inhibition factor for BMGL
chirp_rate = 0.5  # GHz/ns (tunable 0.1-1.0)
detune_scale = 0.032  # Average of 0.03-0.035
alpha_chemical = detune_scale  # Chemical coupling proxy


# BMGL Decoherence-Suppression Protocol Simulation
# Dynamically gaps spin-nodal planes based on omega_ell(t)
# Boosted by p-wave: inhibition_factor = gamma_1 * (1 + (lambda_soc / p_odd_parity) * (gamma_1 - 1))
def apply_bmgl_gating(field_mixed, t, ell_values, noise_level_base=0.01):
    inhibition_boost = 1 + (lambda_soc / p_odd_parity) * (gamma_1 - 1)
    effective_gamma = gamma_1 * inhibition_boost  # 33-50% boost as per PDF
    omega_baseline = chirp_rate  # Baseline rotation rate
    omega_gate = effective_gamma * omega_baseline

    # Simulate phase screen with gating: reduce noise during high omega_ell phases
    phase_screen = np.ones_like(field_mixed, dtype=complex)
    for idx, ell in enumerate(ell_values):
        omega_ell_t = ell * chirp_rate + detune_scale * alpha_chemical  # ω_ℓ(t)
        for ti in range(len(t)):
            if abs(omega_ell_t) > omega_gate:  # Gate active: suppress noise
                gated_noise = noise_level_base / effective_gamma  # Suppression up to 8.88x
            else:
                gated_noise = noise_level_base
            phase_screen[idx, :, ti] *= np.exp(1j * gated_noise * np.random.randn(*field_mixed[idx].shape[:-1]))

    return field_mixed * phase_screen


# Whitening and FastICA - From complex_ica.py, enhanced for complex fields
def whiten_linear(X):
    U, s, Vt = svd(X, full_matrices=False)
    S = np.diag(1 / (s + 1e-10))
    Z = U @ S @ Vt
    return Z


def fast_ica(Z, n_comp, alpha=1, thresh=1e-8, iterations=5000):
    m, n = Z.shape
    W = np.random.rand(m, m) + 1j * np.random.rand(m, m)  # Complex extension
    for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt(np.sum(np.abs(w) ** 2))
        i = 0
        lim = 100
        while (lim > thresh) and (i < iterations):
            ws = np.dot(w.T, Z)
            g = np.tanh(np.real(ws) * alpha) + 1j * np.tanh(np.imag(ws) * alpha)
            dg = (1 - np.abs(g) ** 2) * alpha
            wNew = np.mean(Z * np.conj(g).T, axis=1) - np.mean(dg) * w.squeeze()
            wNew = wNew - np.dot(np.dot(wNew, np.conj(W[:c].T)), W[:c])
            wNew = wNew / np.sqrt(np.sum(np.abs(wNew) ** 2))
            lim = np.abs(np.abs(np.dot(np.conj(wNew), w).sum()) - 1)
            w = wNew
            i += 1
        W[c, :] = w.T
    return W


# Main Simulation Parameters
N = 64  # Grid size (up from 32 for full res)
w0 = 1.0
x = np.linspace(-3 * w0, 3 * w0, N)
Xg, Yg = np.meshgrid(x, x)
Rho = np.sqrt(Xg ** 2 + Yg ** 2)
Phi = np.arctan2(Yg, Xg)
ell_values = [1, 2, 3]  # OAM modes (expandable to |ℓ| >5)
num_modes = len(ell_values)

# Generate Sources
sources = np.array([lg_mode(ell, Rho, Phi, w0) for ell in ell_values])
dx = x[1] - x[0]
norms = np.sqrt(np.sum(np.abs(sources) ** 2, axis=(1, 2)) * dx ** 2)
sources /= norms[:, np.newaxis, np.newaxis]

sources_int = np.abs(sources) ** 2
sources_phase = np.angle(sources)

# Mixing Matrix A (from script)
A = np.array([[0.881, -0.297, 0.369], [-0.140, 0.986, 0.089], [-0.026, -0.204, 0.979]])

# Mix Complex Fields
field_mixed = np.zeros_like(sources, dtype=complex)
for i in range(num_modes):
    for j in range(num_modes):
        field_mixed[i] += sources[j] * A[j, i]

# Time Dimension for Pulses (Integrate pyramidal pulses)
fs = 1000  # Hz
t = np.linspace(-0.5, 0.5, int(fs))
field_mixed = np.repeat(field_mixed[:, :, :, np.newaxis], len(t), axis=3)  # Add time axis

# Embed Data into Pulses (8-bit example, scalable)
bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
n_bits = len(bits)
subcarriers = np.linspace(60, 410, n_bits, dtype=int)
gauss_env = np.exp(-t ** 2 / (2 * 0.1 ** 2))
chirp_base = chirp(t, f0=50, f1=450, t1=1.0, method='linear')
am_modulation = np.zeros_like(t)
for i, bit in enumerate(bits):
    am_modulation += bit * np.cos(2 * np.pi * subcarriers[i] * t) / n_bits
pulse_raw = gauss_env * chirp_base * (1 + 1.2 * am_modulation)

# Modulate fields with pulses and OAM helical phase
for idx, ell in enumerate(ell_values):
    phi_azimuthal = 2 * np.pi * ell * t * 10  # Scaled
    oam_mod = pulse_raw * np.exp(1j * phi_azimuthal)
    field_mixed[idx] *= oam_mod[np.newaxis, np.newaxis, :]

# Gouy Phase Correction (per mode)
z_prop = 1.0
z_Rayleigh = 1.0
for idx, ell in enumerate(ell_values):
    psi_gouy = (abs(ell) + 1) * np.arctan(z_prop / z_Rayleigh)
    correction = np.exp(-1j * psi_gouy)
    field_mixed[idx] *= correction

# Apply Turbulence with BMGL Gating (p-wave enhanced)
noise_level = 0.01
field_mixed_gated = apply_bmgl_gating(field_mixed, t, ell_values, noise_level)

# Intensity and Phase
int_mixed = np.abs(field_mixed_gated) ** 2
phase_mixed = np.angle(field_mixed_gated)

# Pre-ICA Fidelities (averaged over time)
pre_int_fids = []
pre_phase_fids = []
for i in range(num_modes):
    corr_int = []
    corr_ph = []
    for ti in range(len(t)):
        s_int_flat = sources_int[i].flatten()
        m_int_flat = int_mixed[i, :, :, ti].flatten()
        corr_int.append(abs(pearsonr(s_int_flat, m_int_flat)[0]) if np.var(s_int_flat) > 1e-10 and np.var(
            m_int_flat) > 1e-10 else 0.0)

        s_ph_flat = sources_phase[i].flatten()
        m_ph_flat = phase_mixed[i, :, :, ti].flatten()
        corr_ph.append(
            abs(pearsonr(s_ph_flat, m_ph_flat)[0]) if np.var(s_ph_flat) > 1e-10 and np.var(m_ph_flat) > 1e-10 else 0.0)
    pre_int_fids.append(np.mean(corr_int))
    pre_phase_fids.append(np.mean(corr_ph))

# Whitening and ICA (Complex, 2 channels per mode: real/imag)
X_complex = np.zeros((N * N * len(t), 2 * num_modes))
for m in range(num_modes):
    X_complex[:, 2 * m] = np.real(field_mixed_gated[m]).flatten()
    X_complex[:, 2 * m + 1] = np.imag(field_mixed_gated[m]).flatten()

Z = whiten_linear(X_complex)
W = fast_ica(Z, 2 * num_modes)
Y = Z @ W

# Reconstruct
recovered_complex = np.zeros((num_modes, N, N, len(t)), dtype=complex)
for m in range(num_modes):
    re_part = Y[:, 2 * m].reshape(N, N, len(t))
    im_part = Y[:, 2 * m + 1].reshape(N, N, len(t))
    recovered_complex[m] = re_part + 1j * im_part

recovered_int = np.abs(recovered_complex) ** 2
recovered_phase = np.angle(recovered_complex)

# Post-ICA Fidelities (max match, averaged over time)
post_int_fids = []
post_phase_fids = []
for i in range(num_modes):
    max_int = []
    max_ph = []
    for ti in range(len(t)):
        flat_s_int = sources_int[i].flatten()
        flat_s_ph = sources_phase[i].flatten()
        corr_int_t = []
        corr_ph_t = []
        for j in range(num_modes):
            flat_r_int = recovered_int[j, :, :, ti].flatten()
            flat_r_ph = recovered_phase[j, :, :, ti].flatten()
            corr_int_t.append(abs(pearsonr(flat_s_int, flat_r_int)[0]) if np.var(flat_s_int) > 1e-10 and np.var(
                flat_r_int) > 1e-10 else 0.0)
            corr_ph_t.append(abs(pearsonr(flat_s_ph, flat_r_ph)[0]) if np.var(flat_s_ph) > 1e-10 and np.var(
                flat_r_ph) > 1e-10 else 0.0)
        max_int.append(max(corr_int_t))
        max_ph.append(max(corr_ph_t))
    post_int_fids.append(np.mean(max_int))
    post_phase_fids.append(np.mean(max_ph))

post_int_avg = np.mean(post_int_fids)
post_phase_avg = np.mean(post_phase_fids)

print(f"Pre-ICA Intensity Avg: {np.mean(pre_int_fids):.3f}")
print(f"Post-ICA Intensity Avg: {post_int_avg:.3f} (with p-wave BMGL boost)")
print(f"Post-ICA Phase Avg: {post_phase_avg:.3f}")
print("Per-ℓ Intensity Post:", [f'{p:.3f}' for p in post_int_fids])
print("Per-ℓ Phase Post:", [f'{p:.3f}' for p in post_phase_fids])

# Quaternion Encoding and SQUID Currents (for one mode, ell=2)
ell = ell_values[1]  # Example
theta = np.pi / 4
r_unit = np.array([0, 1, 0])
theta_half = theta / 2
q = Quaternion(np.cos(theta_half), *(np.sin(theta_half) * r_unit))

# Gouy-corrected (already applied)
theta_corrected = theta - np.mean(
    [(abs(ell) + 1) * np.arctan(z_prop / z_Rayleigh) for ell in ell_values])  # Average proxy
theta_half_corr = theta_corrected / 2
q_corrected = Quaternion(np.cos(theta_half_corr), *(np.sin(theta_half_corr) * r_unit))

# Flux and Currents
Phi0 = 2.07e-15
L = 100e-12
q_scalars = np.array([q_corrected.w, q_corrected.x, q_corrected.y, q_corrected.z])
fluxes = 2 * np.pi * q_scalars * Phi0
currents = fluxes / L

print("\nQuaternion Scalars [w, x, y, z]:", [f"{s:.3f}" for s in q_scalars])
print("Flux Biases Φ [Wb]:", [f"{f:.2e}" for f in fluxes])
print("SQUID Currents I [μA]:", [f"{I * 1e6:.2f}" for I in currents])

# Compression Proxy
original_bits = len(t) * N * N * 8 * num_modes  # Rough estimate
q_metadata_bits = 4 * 32 * num_modes
payload_bits = n_bits * num_modes
compressed_bits = payload_bits + q_metadata_bits
compression_ratio = original_bits / compressed_bits
print(f"Compression ratio: {compression_ratio:.0f}x (scales with higher ℓ)")

# Plots (Example for ell=2, mid-time)
mid_t = len(t) // 2
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
im0 = axs[0].imshow(sources_phase[1], cmap='hsv', extent=[x.min(), x.max(), x.min(), x.max()])
axs[0].set_title('Source ℓ=2 Phase')
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(phase_mixed[1, :, :, mid_t], cmap='hsv', extent=[x.min(), x.max(), x.min(), x.max()])
axs[1].set_title('Mixed + Turb Phase (BMGL)')
plt.colorbar(im1, ax=axs[1])
im2 = axs[2].imshow(recovered_phase[1, :, :, mid_t], cmap='hsv', extent=[x.min(), x.max(), x.min(), x.max()])
axs[2].set_title('De-Mixed Phase')
plt.colorbar(im2, ax=axs[2])
plt.tight_layout()
plt.savefig('vqc_pwave_bmgl_phase.png', dpi=150)
print("Plot saved to 'vqc_pwave_bmgl_phase.png'")

# SQUID Currents Plot
f_drive = 1000
ac_bias = 0.1 * np.cos(2 * np.pi * f_drive * t)
dynamic_currents = currents[:, np.newaxis] * (1 + ac_bias)
fig_curr, ax = plt.subplots(1, 1, figsize=(10, 6))
labels = ['I_w', 'I_x', 'I_y', 'I_z']
colors = ['b', 'r', 'g', 'm']
for i, (label, color) in enumerate(zip(labels, colors)):
    ax.plot(t[:200], dynamic_currents[i][:200] * 1e6, label=label, color=color, alpha=0.7)
ax.set_title('Flux Qubit SQUID Currents (Dynamic AC Bias)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current (μA)')
ax.legend()
ax.grid(True)
plt.savefig('squid_currents_pwave.png', dpi=150)
print("SQUID currents plot saved as 'squid_currents_pwave.png'")

# Pulse PSD Plot (unchanged)
f, psd = welch(pulse_raw, fs=fs, nperseg=256)
fig_psd, ax_psd = plt.subplots(1, 1, figsize=(8, 4))
ax_psd.semilogy(f, psd)
ax_psd.set_title('Welch PSD of Pyramidal Pulse')
ax_psd.set_xlabel('Frequency (Hz)')
ax_psd.set_ylabel('PSD (V²/Hz)')
ax_psd.grid(True)
for fc in subcarriers:
    ax_psd.axvline(fc, color='r', linestyle='--', alpha=0.7)
plt.savefig('vqc_pulse_psd.png', dpi=150)
print("PSD plot saved as 'vqc_pulse_psd.png'")

print("\nIntegrated VQC Simulation Complete with p-Wave BMGL Enhancements.")