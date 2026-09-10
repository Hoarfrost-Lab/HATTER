"""Embed a fasta with ESM-C, one bare mean-pooled tensor per sequence id.

Drop-in counterpart of fair-esm's `esm/scripts/extract.py ... --include mean`
for the EvolutionaryScale ESM-C models, so CLEAN/HATTER can read either
encoder's cache: fair-esm writes {'mean_representations': {33: tensor}} and
`format_esm` unwraps it; this writes the tensor itself, which `format_esm`
passes through unchanged.

    python esmc_extract.py esmc_600m sequences.fasta out_dir [--batch-tokens N]

Must run under an environment with the EvolutionaryScale `esm` package
(`pip install esm`), NOT fair-esm: both import as `esm`. Mean pooling is over
residue positions only (BOS/EOS excluded), matching fair-esm's "mean".
Skips ids whose .pt already exists, so it is safe to re-run.

Widths: esmc_300m -> 960, esmc_600m -> 1152.
"""
import argparse, os, sys, time

import torch


def read_fasta(path):
    recs, cur_id, cur = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    recs.append((cur_id, "".join(cur)))
                cur_id, cur = line[1:].split()[0], []
            else:
                cur.append(line)
    if cur_id is not None:
        recs.append((cur_id, "".join(cur)))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="esmc_300m or esmc_600m")
    ap.add_argument("fasta")
    ap.add_argument("out_dir")
    ap.add_argument("--batch-tokens", type=int, default=16000,
                    help="max residues per forward batch (sequences are length-sorted)")
    ap.add_argument("--max-len", type=int, default=1022, help="truncate longer sequences")
    a = ap.parse_args()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    os.makedirs(a.out_dir, exist_ok=True)
    recs = [(i, s[: a.max_len]) for i, s in read_fasta(a.fasta)
            if not os.path.exists(os.path.join(a.out_dir, i + ".pt"))]
    if not recs:
        print("  nothing to embed"); return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(a.model).to(dev).eval()
    cfg = LogitsConfig(sequence=False, return_embeddings=True)
    recs.sort(key=lambda r: len(r[1]))
    t0, done = time.time(), 0
    with torch.no_grad():
        i = 0
        while i < len(recs):
            # greedy length-sorted batches under a residue budget
            j, tokens = i, 0
            while j < len(recs) and tokens + len(recs[j][1]) + 2 <= a.batch_tokens:
                tokens += len(recs[j][1]) + 2; j += 1
            j = max(j, i + 1)
            batch = recs[i:j]
            toks = model._tokenize([s for _, s in batch]).to(dev)      # padded [B, L+2]
            out = model(sequence_tokens=toks)
            emb = out.embeddings                                       # [B, L+2, D]
            for k, (sid, seq) in enumerate(batch):
                L = len(seq)
                v = emb[k, 1:L + 1].float().mean(dim=0).cpu()          # residues only
                torch.save(v, os.path.join(a.out_dir, sid + ".pt"))
            done += len(batch); i = j
            if done % 2000 < len(batch) or i == len(recs):
                el = time.time() - t0
                print(f"  {done:,}/{len(recs):,}  {done/el:.1f} seq/s  "
                      f"ETA {(len(recs)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)
    print(f"=== esmc_extract done: {done:,} sequences, dim {v.numel()}, {(time.time()-t0)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
