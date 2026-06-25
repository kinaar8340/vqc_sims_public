# Orbital Braille — VQC Typehead Prototype

Working simulation of the **VQC Typehead / Orbital Braille** embodiment. This is the **standalone layout** of the code in [`kinaar8340/vqc_proto`](https://github.com/kinaar8340/vqc_proto) (integrated layout lives in that repo's `proto/` subdirectory).

The **Vortex Quaternion Conduit (VQC)** multiplexes data into orthogonal OAM modes per DWDM channel, compresses shards via quaternion encoding, and propagates through nested helical beams with BMGL/QEC. Full spec: [VQC Non-Provisional (draft)](https://github.com/kinaar8340/qvpic/blob/main/docs/VQC_NonProvisional_Patent_Application.md).

---

## Demo

![Orbital Braille demo](outputs/orbital_braille_demo.png)

**Payload:** `"I live in Oregon"` · **4 orbs** · **92.9% shard fidelity** after p-wave BMGL turbulence

**Full technical doc:** see the expanded [`proto/README.md`](https://github.com/kinaar8340/vqc_proto/blob/main/proto/README.md) on GitHub.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python run_demo.py --payload "I live in Oregon" --num-orbs 4
.venv/bin/python sweep_orbs.py
.venv/bin/python meta_optimize_orbital.py
.venv/bin/python generate_slm_holograms.py
```

---

## Orb sweep

| Orbs | Fisher-Rao sep | Shard FID | Glyph FID | Verdict |
|------|----------------|-----------|-----------|---------|
| 2 | 0.787 | 0.937 | 0.999 | Cramped alphabet |
| **4** | **0.989** | **0.929** | 0.868 | **Sweet spot** |
| 6 | 1.027 | 0.920 | 0.804 | Harder demux |

### Why 4 orbs?

Near-ideal Fisher-Rao glyph separation (~1 rad), >92% shard fidelity through BMGL, extended Braille analog, and feasible SLM/mechanical hardware mapping. See [proto/README.md on GitHub](https://github.com/kinaar8340/vqc_proto/blob/main/proto/README.md) for full rationale and patent alignment.

---

## Repository

| Repo / branch | URL |
|---------------|-----|
| **vqc_proto** (integrated) | https://github.com/kinaar8340/vqc_proto |
| Standalone backup branch | https://github.com/kinaar8340/vqc_sims_public/tree/vqc_proto-standalone |
| Parent simulations | https://github.com/kinaar8340/vqc_sims_public |

```bash
git clone git@github.com:kinaar8340/vqc_proto.git
cd vqc_proto/proto   # integrated layout
```

---

## License

CC-BY-NC-SA-4.0 with patent restrictions (see parent [LICENSE](https://github.com/kinaar8340/vqc_proto/blob/main/LICENSE)).