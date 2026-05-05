# RFdiffusion3 vs Chroma

Comparative analysis of two generative models for de novo protein design: **RFdiffusion3** (atom-level, Baker Lab / IPD) and **Chroma** (probabilistic, Generate Biomedicines).

## Tasks

| # | Task | Notebook |
|---|------|----------|
| 0 | Setup, env check, dataset prep | `00_setup_and_data.ipynb` |
| 1 | Unconditional generation: efficiency + diversity | `01_unconditional_benchmark.ipynb` |
| 2 | DNA-binding protein design | `02_dna_binder_design.ipynb` |
| 3 | Small-molecule binder design | `03_small_molecule_binder.ipynb` |
| 4 | Enzyme active-site scaffolding | `04_enzyme_scaffolding.ipynb` |
| 5 | Refolding evaluation (ESMFold) | `05_refolding_evaluation.ipynb` |
| 6 | Final aggregation, plots, report | `06_final_analysis.ipynb` |

## Quick start

1. Open any notebook in **Google Colab** (use a GPU runtime: T4 minimum, A100 recommended).
2. Run `00_setup_and_data.ipynb` first — it mounts Drive, installs `rc-foundry` (RFD3 + MPNN) and `generate-chroma`, downloads weights, and fetches benchmark targets.
3. Then run notebooks 01–05 in any order, then 06 to aggregate.
4. All results land in `<Drive>/rfd3-vs-chroma/results/`.

## Models

- **RFdiffusion3** is invoked via the foundry `rfd3 design` CLI with a JSON input spec:
  ```bash
  pip install "rc-foundry[rfd3]"
  foundry install rfd3
  rfd3 design inputs=spec.json out_dir=output/ diffusion_batch_size=8
  ```
- **Chroma** uses the Python API:
  ```python
  from chroma import api, Chroma, conditioners
  api.register_key("YOUR_KEY")
  chroma = Chroma()
  prot = chroma.sample(chain_lengths=[150], steps=200, sde_func='langevin')
  prot.to("out.pdb")
  ```
  Free key at <https://chroma-weights.generatebiomedicines.com/>.
- **LigandMPNN** for sequence design — stable upstream `dauparas/LigandMPNN` repo
  (auto-cloned + weights downloaded by `utils.ensure_ligandmpnn()`).

## Datasets

- **PDB** — protein-protein, protein-DNA, protein-ligand complexes (RCSB)
- **AlphaFold DB** — pre-trained reference structures
- Held-out DNA targets: `7M5W`, `7RTE`, `7N5U` (RFD3 paper §3.3)
- Small-molecule targets: `FAD/7BKC`, `OQO/7V11`, `IAI/5SDV`, `SAM/7C7M` (RFD3 paper §3.4)
- Enzyme reference: `1EUV` (Ulp1 cysteine hydrolase)
- PPI benchmark targets: PD-L1, InsulinR, IL-7Ra, Tie2, IL-2Ra (RFD3 paper §3.2)

## Tools

| Purpose | Tool |
|---------|------|
| Backbone generation | RFdiffusion3 (`rfd3` CLI), Chroma (Python) |
| Sequence design | LigandMPNN / ProteinMPNN (standalone repo, auto-installed) |
| Structure prediction | ESMFold (Colab-friendly) |
| Analysis | Biotite, BioPython, RDKit, SciPy |

## Hardware

- **Free Colab (T4, 15 GB)** — works for all notebooks at default `N_DESIGNS`. ESMFold uses fp16 + chunking.
- **Colab Pro (A100, 40 GB)** — full benchmark; bump `N_DESIGNS` to paper levels (100–400).
- Notebooks degrade gracefully on T4 by default.

## Repository layout

```
.
├── README.md                     ← this file
├── requirements.txt              ← Python deps
├── notebooks/                    ← 7 Colab notebooks
├── scripts/utils.py              ← shared helpers (PDB I/O, RMSD, TM-score, RunRecord)
├── docs/
│   ├── ARCHITECTURE.md           ← design decisions
│   └── TROUBLESHOOTING.md        ← common errors
├── data/                         ← gitignored; populated at runtime
└── results/                      ← gitignored; populated at runtime
```

## Caveats

- **Chroma + DNA** is a baseline only — Chroma has no DNA training data. Use RFD3-NA for real DNA work.
- **Approximate TM-score** in `utils.py` — fine for diversity ranking, not for paper figures. Use TM-align binary for publication.
- **`N_DESIGNS` is reduced** vs. paper (12–25 instead of 100–400). Bump up on A100 if you need statistical power.

## Citation

- Butcher et al. (2025) *De novo Design of All-atom Biomolecular Interactions with RFdiffusion3* (bioRxiv)
- Ingraham et al. (2023) *Illuminating protein space with a programmable generative model.* Nature 623:1070
- Ahern et al. (2025) *Atom level enzyme active site scaffolding using RFdiffusion2* (bioRxiv)

## License

MIT (project code). Underlying models retain their own licenses.
