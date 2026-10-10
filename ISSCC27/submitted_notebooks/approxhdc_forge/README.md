# ApproxHDC-Forge: what does on-chip learning cost in silicon?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mohit14036/sscs-ose-code-a-chip.github.io/blob/approxhdc-forge/ISSCC27/submitted_notebooks/approxhdc_forge/approxhdc_forge.ipynb)

Submission to the IEEE SSCS Code-a-Chip (ISSCC 2027). Everything runs on open-source tools and the SkyWater **sky130** PDK.

Hyperdimensional computing (HDC) classifies by comparing wide binary vectors, and *training is bundling*: the same accumulate the encoder already does. That makes on-chip online learning cheap in principle. This project measures what it costs in actual layouts, and what it buys.

**Start with [`approxhdc_forge.ipynb`](approxhdc_forge.ipynb).** It tells the whole story, with every figure tagged *LIVE* (computed now) or *REPLAYED* (re-plotted from the CSVs in `results/`).

## Results at a glance

All numbers are produced by the scripts below and are in `results/`.

| Question | Answer | Evidence |
|---|---|---|
| What does learning hardware cost? | **1.85x to 6.7x** the area of a matched inference-only chip (median 3.6x), on real sky130 layouts. About 77% of the area scales with the number of stored counter bits. | 34 conventional sign-off-clean layouts, `results/flow_runs.csv` |
| Is approximating the similarity (popcount) unit worthwhile? | Barely: about 2% area at equal accuracy. It is not where the area is. | equal-area study, `figs/day1_equal_area.png` |
| Is approximating the storage worthwhile? | Yes: narrow counters with a stochastic update rule cut area a lot, at an accuracy cost that the search quantifies. | `results/day4_space.csv` (4,260 designs) |
| What does the final chip A reach? | **0.83 accuracy on held-out windows of 9 unseen users** after adapting (0.59 frozen); the larger inference-only chip B reaches 0.74. | `results/final_test.csv` |
| Energy of chip A? | Conventional: 576 nJ per classification, 541 nJ per learning update. **Clock-gated ring: 79 nJ per classification, 76 nJ per learning update** (switching-activity-annotated, typical corner, 50 MHz). | `results/power_*.json` |
| What does clock gating buy? | Gating the ring memory (86-92% of the flip-flops) with the sky130 clock-gate cell cut area **23-28%**, power **4-7x** and energy per operation **77-86%**, with identical function (verified at RTL and gate level). | `results/flow_runs.csv`, `results/power_*.json`, `results/clock_gating_activity.json` |
| Does the search work? | NSGA-II and Bayesian optimization reach the exhaustively enumerated Pareto front ~3x faster than random search. | `results/day4_optimizers.csv` |
| Is the RTL right? | 28/28 configurations (24 conventional + 4 clock-gated) match the model bit for bit on Verilator and again on Icarus (4-state); 7/7 injected bugs caught; gate-level netlists of A and B (conventional) and of A, B and C (clock-gated) match from both flop power-up states. | `results/verification_*.csv`, `results/mutation_test.csv`, `results/gate_level_check.csv` |

**What we do not claim:** a software logistic regression on the same 64 features reaches 0.89 on the same held-out windows, higher than any of our chips. Only the ring memory and the load staging register are clock-gated; the rest of the flip-flops are not. Timing is closed at the typical corner only (slow-corner Fmax is about 42-44 MHz conventional, 46-47 MHz clock-gated). See Section 12 of the notebook.

## The three final chips (full sign-off: Magic DRC, KLayout DRC, LVS, routing DRC all clean)

| Chip | Configuration | Std-cell area, conventional | Std-cell area, clock-gated (`--cg 1`) | Role |
|---|---|---|---|---|
| **A** | D=512, P=32, `satcomp(2)`, bundle-all learning, 2-bit counters | 0.426 mm^2 | **0.309 mm^2** | final chip |
| **B** | D=1024, P=32, `satcomp(2)`, learning off | 0.394 mm^2 | 0.283 mm^2 | matched inference-only chip |
| **C** | D=256, P=32, `satcomp(2)`, bundle-all learning, 2-bit counters | 0.261 mm^2 | 0.201 mm^2 | smallest learning chip |

Chip A's GDS and post-layout netlist (conventional and clock-gated) are in `artifacts/`.

## Repository layout

```
approxhdc_forge.ipynb      the notebook (start here)
colab_flow_test.ipynb      companion notebook: one small design RTL -> GDS in Colab, no Docker
src/
  data.py                  UCI-HAR download (checksum), subject-wise split, feature selection
  hdc_model.py             the bit-accurate golden model (encoding, similarity units, online learning)
  rtlgen.py                Verilog generator (D, P, F, Q, C, similarity, learning mode, counter width, k, seed, clock gating)
  flow_runner.py           config -> LibreLane -> GDS + metrics -> results/flow_runs.csv
  area_model.py            Yosys block-level area estimates (used as surrogate features)
  gls.py                   gate-level simulation of a finished netlist; VCD-annotated power with OpenSTA
  exp_day1.py              algorithm sweeps: accuracy vs D, approximation, samples learned, equal-area study
  exp_day4_model.py        user adaptation, approximation recovery, bit-flip robustness
  exp_day4_search.py       surrogate, enumeration of 4,260 designs, random vs NSGA-II vs BO
  final_eval.py            final numbers on the 9 official test subjects
  prune_flow.py            frees disk from finished flow runs (optional)
tb/                        cocotb testbenches, verification runner, bug-injection test
data/splits.json           the fixed split
results/, figs/            every CSV/JSON and figure the notebook shows
artifacts/                 chip A: GDS (gzip), netlist (gzip), SDC, metrics
versions.txt               exact tool versions;  requirements.txt  Python packages;  LICENSE  Apache-2.0
```

(The `day1`/`day4` prefixes only record the order the work was done in.)

## Running it

**Easiest: Colab.** Open the badge above and run the cells (about 3 minutes live; the heavy results are replayed from `results/`). The companion `colab_flow_test.ipynb` reproduces one small RTL-to-GDS run in Colab without Docker (about 45 minutes; our run matched the Docker result to 0.01% in area).

**Locally.**

```bash
pip install -r requirements.txt          # Python packages
# tools: Docker (LibreLane runs in ghcr.io/librelane/librelane:3.0.14), plus Yosys, Verilator and
# Icarus Verilog (OSS CAD Suite) for simulation; exact versions in versions.txt

python src/exp_day1.py                   # algorithm sweeps + equal-area study      (~6 min, 14 cores)
python src/exp_day4_model.py             # adaptation, recovery, bit flips          (~3 min)
python src/exp_day4_search.py            # surrogate, enumeration, optimizers       (~45 min; cached after)
python src/final_eval.py                 # final numbers on the test subjects       (~15 s)
python tb/run_verif.py                   # 24-config RTL verification (Verilator)   (~25 min)
python tb/mutation_test.py               # injected-bug check                       (~3 min)
python src/flow_runner.py --D 512 --P 32 --variant satcomp --knob 2 --learn 1 --w 2 --k 0 --mode full --render
                                         # one chip, RTL -> GDS                      (~1 h)
#   add  --cg 1  for the clock-gated implementation (about 45 min)
python src/gls.py <flow dir> check       # gate-level match                         (~5 min)
python src/gls.py <flow dir> power       # activity-annotated power                 (~10 min)
```

All random choices are seeded. UCI-HAR is downloaded by URL and checked against a SHA-256.

## License

Apache-2.0, see `LICENSE`. The Nix setup cells in `colab_flow_test.ipynb` are adapted from LibreLane's own notebook (Apache-2.0).
