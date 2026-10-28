"""
Check whether a warhead survives generation when the model starts from a
molecule that already carries it.

This is a different question from scripts/check_warheads.py, which screens
unconditionally generated molecules for warheads they were never given. Here
each seed molecule is encoded to a VAE latent, noised to timestep start_t, and
run back down the reverse diffusion chain, so the output is an analogue of the
seed. The warhead is "preserved" if it still matches in the decoded molecule.

The VAE-only control row matters for reading the result: a seed can lose its
warhead to reconstruction error alone, before diffusion does anything, so the
control is the ceiling the diffusion rows should be compared against.

Writes:
    <out-prefix>_preservation_summary.csv   one row per start_t, plus the control
    <out-prefix>_preservation_detail.csv    one row per (seed, start_t)

Example:
    python scripts/check_warhead_preservation.py --warhead 'C=CC(=O)N'
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger

FILE_PATH = str(Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)

from src.models.ddpm import LatentDiffusion

RDLogger.DisableLog('rdApp.*')

INPUT_DIM = 1024
MLP_DIMS = 2048


def compile_warhead(pattern):
    """Compile the warhead as SMARTS, falling back to SMILES."""
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        query = Chem.MolFromSmiles(pattern)
    if query is None:
        raise SystemExit(f'could not parse warhead {pattern!r}')
    return query


def load_seeds(path, query, n_seeds, smiles_column=None):
    """
    Collect seed molecules that contain the warhead.

    Accepts a .csv (with a smiles column) or a plain one-SMILES-per-line .txt,
    and keeps only molecules that actually match, since a seed without the
    warhead says nothing about preserving it.
    """
    if str(path).endswith('.csv'):
        df = pd.read_csv(path)
        if smiles_column is None:
            smiles_column = next((c for c in ('canonical_smiles', 'smiles') if c in df.columns), None)
            if smiles_column is None:
                raise SystemExit(f'no smiles column in {path}; columns are {list(df.columns)}')
        candidates = df[smiles_column].dropna().tolist()
    else:
        with open(path) as f:
            candidates = [line.strip() for line in f if line.strip()]

    seeds = []
    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(query):
            seeds.append(smiles)
        if len(seeds) >= n_seeds:
            break
    return seeds


def encodable(model, smiles):
    """
    True if the SELFIES encoding fits the VAE's fixed length and vocabulary.

    smiles_to_indices pads up to max_len but does not truncate, and raises on an
    unseen symbol, so longer or out-of-vocabulary molecules have to be dropped.
    """
    try:
        return len(model.smiles_to_indices(smiles)) == model.max_len
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--warhead', default='C=CC(=O)N',
                        help='warhead to track, as SMARTS or SMILES (default: acrylamide)')
    parser.add_argument('--seeds', default=REPO_PATH + '/data/zinc250k/zinc250k_smiles.txt',
                        help='.csv or .txt of candidate seed molecules')
    parser.add_argument('--smiles-column', default=None,
                        help='which column to read if --seeds is a .csv')
    parser.add_argument('--n-seeds', type=int, default=100,
                        help='how many warhead-bearing seeds to use (default: 100)')
    parser.add_argument('--start-t', default='50,100,200,300',
                        help='comma-separated timesteps to noise the seeds to')
    parser.add_argument('--ckpt', default=REPO_PATH + '/model/drugdiff.ckpt')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--seed', type=int, default=0, help='torch seed for the noise')
    parser.add_argument('--out-prefix', default=None,
                        help='output prefix (default: outputs/warhead_<pattern>)')
    args = parser.parse_args()

    start_ts = [int(t) for t in args.start_t.split(',') if t.strip()]
    safe = ''.join(c if c.isalnum() else '_' for c in args.warhead)
    prefix = args.out_prefix or f'{REPO_PATH}/outputs/warhead_{safe}'
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    query = compile_warhead(args.warhead)

    model = LatentDiffusion(input_dim=INPUT_DIM, output_dim=INPUT_DIM, mlp_dims=MLP_DIMS,
                            num_sample_mols=args.n_seeds, vae_ckpt=None,
                            uncond_dm_ckpt=args.ckpt, dataset='zinc250k')
    model.to(args.device)
    model.eval()
    print(f'++++ SAMPLING ON: {model.betas.device}')

    seeds = load_seeds(args.seeds, query, args.n_seeds * 2, args.smiles_column)
    seeds = [s for s in seeds if encodable(model, s)][:args.n_seeds]
    if not seeds:
        raise SystemExit(f'no usable seed molecules containing {args.warhead!r} in {args.seeds}')
    print(f'{len(seeds)} seed molecules containing {args.warhead}')

    canonical_seeds = [Chem.CanonSmiles(s) for s in seeds]

    def evaluate(label, start_t, decoded):
        """Score one batch of decoded molecules against their seeds."""
        rows = []
        for seed, canon, out in zip(seeds, canonical_seeds, decoded):
            mol = Chem.MolFromSmiles(out)
            rows.append({'start_t': start_t,
                         'seed_smiles': seed,
                         'output_smiles': out,
                         'valid': mol is not None,
                         'warhead_preserved': bool(mol is not None and mol.HasSubstructMatch(query)),
                         'molecule_changed': bool(mol is not None and Chem.MolToSmiles(mol) != canon)})
        detail = pd.DataFrame(rows)
        valid = detail[detail.valid]
        summary = {'start_t': start_t,
                   'condition': label,
                   'n_seeds': len(detail),
                   'n_valid': len(valid),
                   'n_preserved': int(valid.warhead_preserved.sum()),
                   'preservation_rate': valid.warhead_preserved.mean() if len(valid) else float('nan'),
                   'n_changed': int(valid.molecule_changed.sum()),
                   'change_rate': valid.molecule_changed.mean() if len(valid) else float('nan')}
        print(f'{label:<28} preserved {summary["n_preserved"]:>3}/{summary["n_valid"]:<3} '
              f'({summary["preservation_rate"]:.0%})   changed {summary["n_changed"]:>3}/{summary["n_valid"]}')
        return detail, summary

    details, summaries = [], []
    with torch.no_grad():
        indices = torch.stack([model.smiles_to_indices(s) for s in seeds]).to(args.device)
        mu = model.vae.encode(indices)[1]

        # control: encode and decode with no diffusion, to isolate VAE loss
        decoded = [model.one_hot_to_smiles(h) for h in torch.exp(model.vae.decode(mu)).cpu()]
        d, s = evaluate('VAE round-trip (control)', -1, decoded)
        details.append(d); summaries.append(s)

        torch.manual_seed(args.seed)
        for start_t in start_ts:
            ts = torch.full((len(mu),), start_t - 1, dtype=torch.long, device=args.device)
            x_T = model.q_sample(x_start=mu, t=ts)
            latents = model.sample(sample_dim=INPUT_DIM, batch_size=len(mu), classifier_scale=0,
                                   x_T=x_T, start_T=start_t, verbose=False)
            decoded = [model.one_hot_to_smiles(h)
                       for h in torch.exp(model.vae.decode(latents.to(args.device))).cpu()]
            d, s = evaluate(f'reverse chain from t={start_t}', start_t, decoded)
            details.append(d); summaries.append(s)

    pd.concat(details).to_csv(f'{prefix}_preservation_detail.csv', index=False)
    pd.DataFrame(summaries).to_csv(f'{prefix}_preservation_summary.csv', index=False)
    print(f'\nwrote {prefix}_preservation_summary.csv')
    print(f'wrote {prefix}_preservation_detail.csv')


if __name__ == '__main__':
    main()
