"""Shared utilities for the RFD3 vs Chroma benchmark."""
from __future__ import annotations
import os, json, time, gzip, hashlib, urllib.request
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ---------- paths ----------
ROOT = Path(os.environ.get("RFD3_CHROMA_ROOT", "/content/rfd3-vs-chroma"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
for p in (DATA, RESULTS):
    p.mkdir(parents=True, exist_ok=True)


# ---------- timing ----------
@contextmanager
def timer(name: str, store: Optional[dict] = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{name}] {dt:.2f}s")
    if store is not None:
        store[name] = dt


# ---------- GPU info ----------
def gpu_info() -> dict:
    info = {"cuda": False, "name": None, "mem_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return info


def free_gpu():
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------- PDB fetch ----------
PDB_URL = "https://files.rcsb.org/download/{pid}.pdb"

def fetch_pdb(pdb_id: str, dest: Path | None = None) -> Path:
    pdb_id = pdb_id.lower()
    dest = dest or DATA / "pdb" / f"{pdb_id}.pdb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(PDB_URL.format(pid=pdb_id), dest)
    return dest


# ---------- structure I/O ----------
def load_ca_coords(pdb_path: Path, chain: str | None = None) -> np.ndarray:
    """Return Cα coordinates as (N, 3) numpy array."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if chain and line[21] != chain:
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords, dtype=np.float32)


def write_pdb_from_ca(coords: np.ndarray, out: Path, seq: str | None = None) -> Path:
    """Write a minimal Cα-only PDB. Useful for placeholder backbones."""
    out.parent.mkdir(parents=True, exist_ok=True)
    aa3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN",
           "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
           "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}
    seq = seq or "A" * len(coords)
    with open(out, "w") as f:
        for i, (xyz, aa) in enumerate(zip(coords, seq), 1):
            r = aa3.get(aa, "ALA")
            f.write(f"ATOM  {i:5d}  CA  {r} A{i:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    return out


# ---------- metrics ----------
def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition (Kabsch)."""
    assert a.shape == b.shape
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    h = ac.T @ bc
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    diff = ac @ r.T - bc
    return float(np.sqrt((diff ** 2).sum() / len(a)))


def tm_score(coords_a: np.ndarray, coords_b: np.ndarray, l_target: int | None = None) -> float:
    """Approximate TM-score (length-normalized). For exact values use TM-align."""
    n = min(len(coords_a), len(coords_b))
    a, b = coords_a[:n], coords_b[:n]
    ac = a - a.mean(0); bc = b - b.mean(0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    di = np.linalg.norm(ac @ r.T - bc, axis=1)
    L = l_target or n
    # d0 must stay positive even for short proteins
    d0 = max(1.24 * (max(L, 19) - 15) ** (1 / 3) - 1.8, 0.5)
    return float(np.mean(1 / (1 + (di / d0) ** 2)))


def radius_of_gyration(coords: np.ndarray) -> float:
    c = coords - coords.mean(0)
    return float(np.sqrt((c ** 2).sum(1).mean()))


# ---------- bookkeeping ----------
@dataclass
class RunRecord:
    model: str
    task: str
    target: str
    length: int
    n_designs: int
    seconds: float
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def append_record(rec: RunRecord, fname: str = "runs.jsonl"):
    p = RESULTS / fname
    with open(p, "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")


def load_records(fname: str = "runs.jsonl") -> list[dict]:
    p = RESULTS / fname
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------- pip installer (idempotent) ----------
def pip_install(*pkgs: str, quiet: bool = True):
    import subprocess, sys
    flags = ["-q"] if quiet else []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *flags, *pkgs])


# ---------- RFD3 spec helpers ----------
def rfd3_run(spec: dict, out_dir: Path, diffusion_batch_size: int = 8,
             n_batches: int = 1, num_timesteps: int = 200,
             extra_args: list[str] | None = None,
             timeout: int | None = None) -> tuple[bool, str, float]:
    """Write the spec to JSON and run `rfd3 design`. Returns (ok, stderr_tail, seconds)."""
    import subprocess, time, json as _json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "spec.json"
    spec_path.write_text(_json.dumps(spec, indent=2))
    cmd = ["rfd3", "design",
           f"inputs={spec_path}",
           f"out_dir={out_dir}",
           f"diffusion_batch_size={diffusion_batch_size}",
           f"n_batches={n_batches}",
           f"inference_sampler.num_timesteps={num_timesteps}",
           "skip_existing=False"]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        dt = time.perf_counter() - t0
        return (r.returncode == 0, r.stderr[-500:] if r.stderr else "", dt)
    except subprocess.TimeoutExpired:
        return (False, "TIMEOUT", time.perf_counter() - t0)
    except FileNotFoundError:
        return (False, "rfd3 CLI not on PATH", 0.0)


# ---------- LigandMPNN ----------
LIGANDMPNN_DIR = Path("/content/LigandMPNN")
LIGANDMPNN_REPO = "https://github.com/dauparas/LigandMPNN.git"

def ensure_ligandmpnn() -> bool:
    """Clone LigandMPNN + download default weights. Idempotent."""
    import subprocess
    if not LIGANDMPNN_DIR.exists():
        try:
            subprocess.run(["git", "clone", "--depth", "1", LIGANDMPNN_REPO,
                            str(LIGANDMPNN_DIR)], check=True, capture_output=True)
        except Exception:
            return False
    weights = LIGANDMPNN_DIR / "model_params"
    if not weights.exists() or not any(weights.glob("*.pt")):
        sh = LIGANDMPNN_DIR / "get_model_params.sh"
        if sh.exists():
            subprocess.run(["bash", str(sh), str(weights)], capture_output=True)
    return (weights.exists() and any(weights.glob("*.pt")))

def run_ligandmpnn(pdb_path: Path, out_dir: Path, n_seqs: int = 4,
                   model_type: str = "protein_mpnn",
                   checkpoint: str = "proteinmpnn_v_48_020.pt") -> bool:
    """Run LigandMPNN/ProteinMPNN on a single PDB. Returns success bool."""
    import subprocess
    if not LIGANDMPNN_DIR.exists():
        if not ensure_ligandmpnn():
            return False
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = LIGANDMPNN_DIR / "model_params" / checkpoint
    if not ckpt.exists():
        return False
    cmd = ["python", str(LIGANDMPNN_DIR / "run.py"),
           "--model_type", model_type,
           f"--checkpoint_{model_type}", str(ckpt),
           "--pdb_path", str(pdb_path),
           "--out_folder", str(out_dir),
           "--number_of_batches", "1",
           "--batch_size", str(n_seqs),
           "--temperature", "0.1",
           "--seed", "37"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0
