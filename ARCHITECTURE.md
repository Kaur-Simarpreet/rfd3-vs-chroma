# Architecture Notes

## Why these comparisons

| Axis | RFdiffusion3 | Chroma |
|---|---|---|
| Representation | All-atom (atom14 + virtual atoms) | Cα + design network for sidechains |
| Conditioning | Motif (indexed/unindexed), hotspots, RASA, H-bonds, symmetry | Substructure, shape, symmetry, classifier guidance, text |
| Sequence design | LigandMPNN (foundry `mpnn`) | Built-in design network |
| Architecture | UNet + sparse SL2 attention | GNN with sub-quadratic random graph |
| Atom-level ligand | Yes (native) | Indirect (via shape conditioner) |
| DNA / NA | Yes (native; RFD3-NA preferred) | No |

## Where the comparison is *not* apples-to-apples

- **DNA (nb 02)**: Chroma never saw DNA in training. We use `ProClassConditioner('cath','3')` as a baseline. Expect Chroma to lose; this is a sanity check only.
- **Small molecules (nb 03)**: RFD3 has native ligand tokens via `ligand` field in JSON spec. Chroma uses `ShapeConditioner` over the ligand atom cloud — closer fight, still favours RFD3.
- **Enzymes (nb 04)**: RFD3 supports unindexed atomic motif scaffolding (`select_unindexed_motif`). Chroma's `SubstructureConditioner` is backbone-only, indexed only. Expect RFD3 to dominate the motif-recapitulation test.
- **Unconditional (nb 01)**: This is the cleanest comparison — both models do this natively.

## Notebook chain

```
00_setup ──┬──> 01_uncond ──┐
           ├──> 02_dna     ─┼──> 05_refold ──> 06_final
           ├──> 03_sm      ─┤
           └──> 04_enzyme  ─┘
```

Each downstream notebook depends only on `00_setup` having run; `05_refold` consumes outputs from 01/03/04. `06_final` aggregates everything in `results/`.

## Real install / invocation

### RFdiffusion3 (foundry)

```bash
pip install "rc-foundry[rfd3]"
foundry install rfd3   # downloads checkpoints
rfd3 design inputs=spec.json out_dir=output/ diffusion_batch_size=8
```

JSON spec format (verified against the foundry input docs):
```json
{
  "design_name": {
    "input": "/path/to/input.pdb",
    "contig": "A1-150",
    "length": "150-150",
    "ligand": "FAD",
    "unindex": "A580,A531,A517",
    "select_fixed_atoms": {"A580": "TIP", "A531": "TIP", "A517": "TIP"}
  }
}
```

**`num_designs` is NOT a JSON spec field** — control design count via the CLI:
- `diffusion_batch_size=8` (default 8): designs per batch
- `n_batches=1` (default 1): number of batches

The helper `utils.rfd3_run(spec, out_dir, diffusion_batch_size=N, ...)` writes the
spec to JSON and shells out for you.

### LigandMPNN (standalone, stable)

We use the upstream `dauparas/LigandMPNN` repo rather than foundry's `mpnn`
because the latter's API is still under active development. The helper
`utils.run_ligandmpnn(pdb, out_dir, model_type='protein_mpnn'|'ligand_mpnn')`
clones the repo + downloads weights on first call, then runs sequence design.

### Chroma

```bash
pip install generate-chroma
```

```python
from chroma import api, Chroma, conditioners
api.register_key("YOUR_KEY")
chroma = Chroma()
protein = chroma.sample(chain_lengths=[150], steps=200, sde_func='langevin')
protein.to("out.pdb")
```

Conditioner construction patterns:

```python
# ShapeConditioner - takes X_target (numpy) + noise_schedule
cond = conditioners.ShapeConditioner(
    X_target=ligand_coords,
    noise_schedule=chroma.backbone_network.noise_perturb.noise_schedule,
)

# SubstructureConditioner - takes a full-length Protein with motif residues
# placed at known positions; selection picks them out
cond = conditioners.SubstructureConditioner(
    protein=full_length_protein,
    backbone_model=chroma.backbone_network,
    selection="A12,A34,A56",  # contig-style residue selection
    rg=False,
)

# ProClassConditioner - CATH/Pfam class guidance
cond = conditioners.ProClassConditioner(
    label='cath', value='3', model='named:public', weight=5.0,
)
```

## Design decisions

- **ESMFold over AF3** for refolding — AF3 needs a separate API key. ESMFold runs on Colab T4 with chunking. Trade-off: no MSA, slightly lower confidence on novel folds.
- **Approximate TM-score** in `utils.py` — for in-notebook diversity. Use the `TM-align` binary for publication numbers.
- **Reduced N_DESIGNS** vs. paper — 12–25 instead of 100–400. Good enough for trends; bump up on A100.
- **No FoldSeek by default** — heavy install. Add for PDB-novelty quantification.

## Compute budget (single A100, 40 GB)

| Notebook | Wall time |
|---|---|
| 00 setup | 15 min (downloads dominate) |
| 01 uncond | 30 min |
| 02 DNA | 1 h |
| 03 small-mol | 1 h |
| 04 enzyme | 45 min |
| 05 refold | 2 h |
| 06 final | 2 min |

T4 multiplies these by roughly 3–4×.
