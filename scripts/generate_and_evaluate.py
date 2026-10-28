"""
Unconditional generation with the ZINC250k DrugDiff checkpoint, plus evaluation.

Writes three .csv files next to each other in outputs/:
    <prefix>_smiles.csv              one row per generated molecule, SMILES only
    <prefix>_per_molecule.csv        one row per molecule with all RDKit properties
    <prefix>_set_level.csv           one row per set-level metric (validity, uniqueness, ...)

Example:
    python scripts/generate_and_evaluate.py --num-mols 1000
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

FILE_PATH = str(Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)

from src.models.ddpm import LatentDiffusion
from src.models.components.evaluation import Evaluator

RDLogger.DisableLog('rdApp.*')

INPUT_DIM = 1024   # do not change: latent size of the trained VAE
MLP_DIMS = 2048    # do not change: width of the trained denoiser


################################################################################
# metrics
#
# canonicalize/remove_duplicates follow GuacaMol; the fingerprint and Tanimoto
# helpers used by internal diversity follow MOSES. Both are the definitions used
# in notebooks/evaluation_without_guidance.ipynb, kept here so a generation run
# reports its own numbers.
################################################################################

def canonicalize(smiles, include_stereocenters=True):
    """Converts a SMILES string to its RDKit canonical form.

    Args:
        smiles: SMILES string to canonicalize.
        include_stereocenters: Whether to keep stereochemistry in the output.

    Returns:
        The canonical SMILES, or None if the molecule does not parse.

    Example:
        >>> canonicalize('OCC')
        'CCO'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    except Exception:
        return None


def remove_duplicates(list_with_duplicates):
    """Drops duplicates, keeping the first occurrence and the original order.

    Args:
        list_with_duplicates: List of hashable items.

    Returns:
        A new list with each item once.

    Example:
        >>> remove_duplicates(['a', 'b', 'a'])
        ['a', 'b']
    """
    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
    return unique_list


def canonicalize_list(smiles_list, include_stereocenters=True):
    """Canonicalizes a list of SMILES, dropping unparseable ones and duplicates.

    Args:
        smiles_list: SMILES strings to process.
        include_stereocenters: Whether to keep stereochemistry.

    Returns:
        Unique canonical SMILES, in order of first appearance.
    """
    canonical = [canonicalize(s, include_stereocenters) for s in smiles_list]
    return remove_duplicates([s for s in canonical if s is not None])


def fingerprint(smiles, fp_type='morgan', morgan__r=2, morgan__n=1024):
    """Computes a fingerprint bit vector for one molecule.

    Args:
        smiles: SMILES string.
        fp_type: ``'morgan'`` or ``'maccs'``.
        morgan__r: Morgan radius (used only for ``'morgan'``).
        morgan__n: Morgan bit-vector length (used only for ``'morgan'``).

    Returns:
        A uint8 array (length 166 for MACCS, ``morgan__n`` for Morgan), or None
        if the molecule does not parse or sanitize.

    Raises:
        ValueError: If ``fp_type`` is not recognised.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    if fp_type == 'maccs':
        keys = np.array(MACCSkeys.GenMACCSKeys(mol).GetOnBits())
        fp = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fp[keys - 1] = 1  # drop the 0-th key, which is always zero
        return fp
    elif fp_type == 'morgan':
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, morgan__r, nBits=morgan__n)
        return np.asarray(bits, dtype='uint8')
    raise ValueError(f'Unknown fingerprint type {fp_type}')


def fingerprints(smiles_list, **kwargs):
    """Stacks the fingerprints of a list of SMILES, skipping the ones that fail.

    Args:
        smiles_list: SMILES strings.
        **kwargs: Passed to ``fingerprint`` (``fp_type``, ``morgan__r``,
            ``morgan__n``).

    Returns:
        A 2D uint8 array with one row per molecule that fingerprinted
        successfully.
    """
    fps = [fp for fp in map(partial(fingerprint, **kwargs), smiles_list) if fp is not None]
    return np.vstack(fps)


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', device='cpu', p=1):
    """Averages, over gen_vecs, each molecule's aggregated Tanimoto similarity to stock_vecs.

    Args:
        stock_vecs: Binary fingerprint matrix of the reference set, one row per
            molecule.
        gen_vecs: Binary fingerprint matrix of the generated set.
        batch_size: Number of rows of each matrix processed at a time.
        agg: ``'max'`` for the nearest neighbour or ``'mean'`` for the average
            similarity to the whole reference set.
        device: Torch device for the matrix products.
        p: Power mean exponent applied to the similarities (1 is a plain mean).

    Returns:
        The mean aggregated similarity as a float.

    Raises:
        AssertionError: If ``agg`` is not ``'max'`` or ``'mean'``.
    """
    assert agg in ['max', 'mean'], 'Can aggregate only max or mean'
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = agg_tanimoto ** (1 / p)
    return np.mean(agg_tanimoto)


def internal_diversity(smiles, device='cpu', fp_type='morgan', p=1):
    """Computes internal diversity as 1 minus the mean pairwise Tanimoto similarity.

    Args:
        smiles: SMILES strings of the set.
        device: Torch device for the similarity computation.
        fp_type: Fingerprint type, ``'morgan'`` or ``'maccs'``.
        p: Power mean exponent (1 is a plain mean).

    Returns:
        Diversity in [0, 1]; higher means the molecules are less alike.
    """
    gen_fps = fingerprints(smiles, fp_type=fp_type)
    return 1 - average_agg_tanimoto(gen_fps, gen_fps, agg='mean', device=device, p=p)


def has_radical(smiles):
    """Checks whether any atom carries unpaired electrons.

    Args:
        smiles: SMILES string.

    Returns:
        True if any atom is a radical, False if none is, and None if the molecule
        does not parse.

    Example:
        >>> has_radical('C[CH2]')
        True
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())


################################################################################
# generation
################################################################################

def generate(num_mols, ckpt, device, seed):
    """Samples molecules unconditionally from the DrugDiff checkpoint.

    Args:
        num_mols: Number of molecules to sample.
        ckpt: Path to the DrugDiff checkpoint.
        device: Torch device to sample on (``'cpu'``, ``'cuda'`` or ``'mps'``).
        seed: Random seed, applied with ``pytorch_lightning.seed_everything``.

    Returns:
        ``num_mols`` raw SMILES strings, not canonicalized or validity-filtered.
    """
    import pytorch_lightning as pl
    pl.seed_everything(seed)

    model = LatentDiffusion(input_dim=INPUT_DIM,
                            output_dim=INPUT_DIM,
                            mlp_dims=MLP_DIMS,
                            num_sample_mols=num_mols,
                            vae_ckpt=None,
                            uncond_dm_ckpt=ckpt,
                            dataset='zinc250k')
    model.to(device)
    model.eval()
    # the device printed during init only reflects the cuda check, so report the
    # device the sampling loop will actually use (it reads it off self.betas)
    print(f'++++ SAMPLING ON: {model.betas.device}')

    # no classifiers -> unconditional, so classifier_scale is unused
    latents = model.sample(sample_dim=INPUT_DIM, batch_size=num_mols, classifier_scale=0)
    one_hots = torch.exp(model.vae.decode(latents.to(device)))
    return [model.one_hot_to_smiles(hot) for hot in one_hots.cpu()]


################################################################################
# main
################################################################################

def main():
    """Generates molecules (or reuses a saved .csv), computes metrics, and writes the output files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-mols', type=int, default=1000,
                        help='how many molecules to generate (default: 1000)')
    parser.add_argument('--ckpt', default=REPO_PATH + '/model/drugdiff.ckpt',
                        help='DrugDiff checkpoint trained on ZINC250k')
    parser.add_argument('--train-smiles', default=REPO_PATH + '/data/zinc250k/zinc250k_smiles.txt',
                        help='training SMILES, used as the reference set for novelty')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--seed', type=int, default=675332,
                        help='seed (default: the one used in apply_guidance)')
    parser.add_argument('--out-prefix', default=None,
                        help='output path prefix (default: outputs/unconditional_<num-mols>)')
    parser.add_argument('--reuse-smiles', default=None,
                        help='recompute the metrics from an existing <prefix>_smiles.csv '
                             'instead of sampling again')
    args = parser.parse_args()

    prefix = args.out_prefix or f'{REPO_PATH}/outputs/unconditional_{args.num_mols}'
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_smiles:
        print(f'reusing molecules from {args.reuse_smiles}')
        smiles = pd.read_csv(args.reuse_smiles).smiles.tolist()
    else:
        smiles = generate(args.num_mols, args.ckpt, args.device, args.seed)

    # ---------------------------------------------------------------- per molecule
    # A generated SELFIES always decodes to *a* SMILES string, but it may not be a
    # parseable molecule, so validity is the share that RDKit accepts.
    canonical = [canonicalize(s) for s in smiles]
    valid_mask = [c is not None for c in canonical]
    valid_smiles = [c for c in canonical if c is not None]
    unique_smiles = remove_duplicates(valid_smiles)

    # The VAE decoder often emits radical carbons, which parse fine but are not
    # real compounds, so also record the repo's cleaned form (clean_smiles
    # hydrogenates radicals and standardizes; it returns 'FAULTY' on failure).
    radical_mask = [has_radical(c) if c is not None else None for c in canonical]
    cleaned = [LatentDiffusion.clean_smiles(c) if c is not None else None for c in canonical]

    df = pd.DataFrame({'mol_id': range(len(smiles)),
                       'smiles': smiles,
                       'canonical_smiles': canonical,
                       'cleaned_smiles': cleaned,
                       'valid': valid_mask,
                       'has_radical': radical_mask})
    df.to_csv(f'{prefix}_smiles.csv', index=False)
    print(f'wrote {prefix}_smiles.csv')

    # properties are only defined for molecules RDKit can parse
    evaluator = Evaluator(max_len=72, symbol_to_idx=None, dataset='zinc250k')
    props = evaluator.computeProperties(valid_smiles)
    props.insert(0, 'has_radical', [r for r in radical_mask if r is not None])
    props.insert(0, 'canonical_smiles', valid_smiles)
    props.insert(0, 'mol_id', [i for i, ok in enumerate(valid_mask) if ok])
    props['is_duplicate'] = props.canonical_smiles.duplicated(keep='first')
    props.to_csv(f'{prefix}_per_molecule.csv', index=False)
    print(f'wrote {prefix}_per_molecule.csv')

    # ---------------------------------------------------------------- set level
    print('loading training set for novelty')
    with open(args.train_smiles) as f:
        train = [line.strip() for line in f if line.strip()]
    train_canonical = set(canonicalize_list(train, include_stereocenters=False))

    # novelty and uniqueness are computed without stereochemistry, as in GuacaMol
    unique_nostereo = set(canonicalize_list(smiles, include_stereocenters=False))
    novel = unique_nostereo - train_canonical

    rows = [
        ('n_generated', len(smiles), 'count',
         'molecules sampled from the model'),
        ('n_valid', len(valid_smiles), 'count',
         'generated molecules RDKit can parse'),
        ('n_unique', len(unique_smiles), 'count',
         'distinct canonical SMILES among the valid molecules'),
        ('validity', len(valid_smiles) / len(smiles), 'n_generated',
         'fraction of sampled molecules that are chemically valid'),
        ('uniqueness', len(unique_smiles) / max(len(valid_smiles), 1), 'n_valid',
         'distinct canonical SMILES / valid molecules'),
        ('novelty', len(novel) / max(len(unique_nostereo), 1), 'n_unique',
         'unique molecules absent from the ZINC250k training set / unique molecules'),
        ('internal_diversity', internal_diversity(unique_smiles), 'n_unique',
         '1 - mean pairwise Morgan Tanimoto similarity within the unique set'),
        ('n_with_radical', sum(1 for r in radical_mask if r), 'count',
         'valid molecules carrying unpaired electrons (repaired by clean_smiles)'),
        ('radical_rate', sum(1 for r in radical_mask if r) / max(len(valid_smiles), 1), 'n_valid',
         'share of valid molecules that are radicals rather than real compounds'),
        ('n_faulty_after_cleaning', sum(1 for c in cleaned if c == 'FAULTY'), 'count',
         'molecules that clean_smiles could not standardize'),
    ]
    set_level = pd.DataFrame(rows, columns=['metric', 'value', 'denominator', 'description'])

    # distribution summaries of the per-molecule properties, over unique molecules
    unique_props = props[~props.is_duplicate]
    summary = []
    for col in props.columns:
        if col in ('mol_id', 'canonical_smiles', 'is_duplicate'):
            continue
        values = unique_props[col].astype(float)
        for stat in ('mean', 'std', 'min', 'median', 'max'):
            summary.append((f'{col}_{stat}', getattr(values, stat)(), 'n_unique',
                            f'{stat} of {col} over unique valid molecules'))
    set_level = pd.concat([set_level, pd.DataFrame(
        summary, columns=['metric', 'value', 'denominator', 'description'])])

    set_level.to_csv(f'{prefix}_set_level.csv', index=False)
    print(f'wrote {prefix}_set_level.csv')
    print()
    print(set_level.head(7).to_string(index=False))


if __name__ == '__main__':
    main()
