# Troubleshooting

### `CUDA out of memory` on T4

- ESMFold (nb 05): already runs in fp16 with `chunk_size=64`. For long sequences try `fold_model.set_chunk_size(32)`.
- Chroma: lower `steps=200 → 100`, drop conditioner weights.
- RFD3: reduce `num_designs` per JSON spec; restart the Colab runtime between runs.

### Chroma API key fails

Chroma weights are gated. Get a free key at <https://chroma-weights.generatebiomedicines.com/> and paste it when notebook 0 prompts (`getpass`).

### `rfd3` / `mpnn` command not found after install

The `rc-foundry` package installs CLI scripts into the venv's `bin/`. On Colab this is on PATH automatically. If not:

```bash
pip install --force-reinstall "rc-foundry[rfd3,mpnn]"
which rfd3
```

If still missing, clone the repo manually:

```bash
git clone https://github.com/RosettaCommons/foundry.git
cd foundry && pip install -e ".[rfd3,mpnn]"
```

### `foundry install rfd3` fails

Retry with `--force`. Default checkpoint dir is `~/.foundry/checkpoints`. To put weights in Drive (survives runtime restarts):

```bash
foundry install rfd3 --checkpoint-dir /content/drive/MyDrive/checkpoints
export FOUNDRY_CHECKPOINT_DIRS=/content/drive/MyDrive/checkpoints
```

### "No DNA atoms found" in nb 02

Some older PDBs use single-letter nucleotide names (`A`, `T`, `G`, `C`) which clash with amino acids. Inspect:

```python
import biotite.structure.io.pdb as bpdb
arr = bpdb.PDBFile.read('data/pdb/7m5w.pdb').get_structure(model=1)
print(set(arr.res_name))
```

If you see `{'A', 'T', 'G', 'C'}` only, extend `DNA_RESNAMES` in nb 02.

### ESMFold predicts garbage on >700 residues

Known limitation. Truncate to a domain, switch to ColabFold, or use the AlphaFold3 server.

### Notebook 06 shows "n/a" everywhere

You haven't run 01–05 yet, or `RFD3_CHROMA_ROOT` env var differs across notebooks. Verify:

```python
import os; print(os.environ.get('RFD3_CHROMA_ROOT'))
```

It should match across all notebooks (default: `/content/drive/MyDrive/rfd3-vs-chroma`).

### Conditioner names

Chroma module exposes (current main):

```python
from chroma import conditioners
# SubstructureConditioner   — fix coordinates of a sub-region
# SubsequenceConditioner    — fix sequence of a sub-region
# ShapeConditioner          — bias toward a target point cloud
# ProClassConditioner       — CATH / pfam / fold class guidance
# ProCapConditioner         — natural-language captioning
# SymmetryConditioner       — Cn / Dn / T / O / I symmetry
```

For coordinate scaffolding use **`SubstructureConditioner`** (not `SubsequenceConditioner`).

### `Protein.from_PDB` fails

Chroma's `Protein` requires standard residues only. Strip HETATMs:

```python
import biotite.structure.io.pdb as bpdb
arr = bpdb.PDBFile.read(pdb).get_structure(model=1)
arr = arr[~arr.hetero]
bpdb.PDBFile().set_structure(arr).write(clean_pdb)
```

### MPNN output structure differs

Foundry `mpnn` writes:
```
<out_folder>/seqs/<input_stem>.fa
<out_folder>/packed/<input_stem>_packed.pdb   (LigandMPNN only)
```
This is what nb 02 / 05 expect.
