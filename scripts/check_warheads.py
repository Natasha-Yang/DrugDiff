"""
Check generated molecules for covalent-warhead substructures.

Reads a file of warhead patterns (one SMARTS or SMILES per line) and a .csv of
generated molecules, then reports every molecule/warhead match.

Patterns are compiled as SMARTS, falling back to SMILES. Note that compiling a
plain SMILES as SMARTS matches more loosely than the drawn fragment does: atoms
carry no implicit-hydrogen requirement, so e.g. "C=CC(=O)N" hits substituted
acrylamides too. That is usually what you want when screening for a warhead.

Writes two .csv files:
    <prefix>_warhead_hits.csv      one row per (molecule, warhead) match
    <prefix>_warhead_summary.csv   one row per warhead, with how many molecules hit

Example:
    python scripts/check_warheads.py --molecules outputs/unconditional_1000_smiles.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

FILE_PATH = str(Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)

RDLogger.DisableLog('rdApp.*')


def describe_match(mol, match, reference):
    """
    Describe how a matched substructure compares with the warhead as drawn.

    Because a plain SMILES compiled as SMARTS puts no constraint on hydrogens,
    a match may be a decorated version of the drawn fragment. Comparing the
    hydrogen count of each matched atom against the reference fragment says
    which it is. Note that a warhead is usually drawn with placeholder methyls
    standing in for the rest of the molecule, so substitution is normal and is
    not by itself a sign of a damaged warhead; radicals in or next to the match
    are.

    Returns (n_substituted, radical_in_match, radical_adjacent).
    """
    n_substituted = None
    if reference is not None and reference.GetNumAtoms() == len(match):
        n_substituted = 0
        for ref_atom, mol_idx in zip(reference.GetAtoms(), match):
            if mol.GetAtomWithIdx(mol_idx).GetTotalNumHs() < ref_atom.GetTotalNumHs():
                n_substituted += 1

    matched = set(match)
    radical_in_match = any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() > 0 for i in match)
    radical_adjacent = any(
        nbr.GetNumRadicalElectrons() > 0
        for i in match for nbr in mol.GetAtomWithIdx(i).GetNeighbors()
        if nbr.GetIdx() not in matched)
    return n_substituted, radical_in_match, radical_adjacent


def load_warheads(path):
    """
    Parse the warhead file into a list of (line_no, pattern, query_mol, how, reference).

    'reference' is the pattern read as a plain molecule, used to tell an intact
    warhead from a substituted one; it is None for patterns that are SMARTS-only.

    Duplicated patterns are collapsed onto their first occurrence, and patterns
    that neither the SMARTS nor the SMILES parser accepts are reported and
    skipped.
    """
    warheads = []
    seen = {}
    skipped = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            pattern = line.strip()
            if not pattern or pattern.startswith('#'):
                continue
            if pattern in seen:
                print(f'  line {line_no}: duplicate of line {seen[pattern]}, skipping')
                continue
            query = Chem.MolFromSmarts(pattern)
            how = 'smarts'
            if query is None:
                query = Chem.MolFromSmiles(pattern)
                how = 'smiles'
            if query is None:
                skipped.append((line_no, pattern))
                continue
            seen[pattern] = line_no
            warheads.append((line_no, pattern, query, how, Chem.MolFromSmiles(pattern)))

    for line_no, pattern in skipped:
        print(f'  line {line_no}: could not parse {pattern!r}, skipping')
    return warheads


def load_molecules(path, column=None):
    """Read the molecules .csv and return (mol_id, smiles, mol) for parseable rows."""
    df = pd.read_csv(path)
    if column is None:
        for candidate in ('canonical_smiles', 'smiles'):
            if candidate in df.columns:
                column = candidate
                break
        else:
            raise SystemExit(f'no smiles column in {path}; columns are {list(df.columns)}')

    ids = df['mol_id'] if 'mol_id' in df.columns else df.index
    molecules = []
    unparseable = 0
    for mol_id, smiles in zip(ids, df[column]):
        if not isinstance(smiles, str):
            unparseable += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unparseable += 1
            continue
        molecules.append((mol_id, smiles, mol))
    return molecules, column, unparseable


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--molecules', default=REPO_PATH + '/outputs/unconditional_1000_smiles.csv',
                        help='.csv of generated molecules')
    parser.add_argument('--smiles-column', default=None,
                        help='which column to read (default: canonical_smiles, else smiles)')
    parser.add_argument('--warheads', default=REPO_PATH + '/warheads.txt',
                        help='warhead patterns, one SMARTS/SMILES per line')
    parser.add_argument('--out-prefix', default=None,
                        help='output prefix (default: alongside the molecules file)')
    args = parser.parse_args()

    prefix = args.out_prefix
    if prefix is None:
        mol_path = Path(args.molecules)
        prefix = str(mol_path.parent / mol_path.stem.replace('_smiles', ''))

    print(f'loading warheads from {args.warheads}')
    warheads = load_warheads(args.warheads)
    print(f'  {len(warheads)} distinct patterns')

    molecules, column, unparseable = load_molecules(args.molecules, args.smiles_column)
    print(f'loaded {len(molecules)} molecules from {args.molecules} (column {column!r})')
    if unparseable:
        print(f'  skipped {unparseable} rows RDKit could not parse')

    hits = []
    for line_no, pattern, query, how, reference in warheads:
        for mol_id, smiles, mol in molecules:
            matches = mol.GetSubstructMatches(query)
            if not matches:
                continue
            # describe the best (least modified) of this warhead's matches
            described = [describe_match(mol, match, reference) for match in matches]
            n_sub = [d[0] for d in described if d[0] is not None]
            hits.append({'mol_id': mol_id,
                         'smiles': smiles,
                         'warhead_line': line_no,
                         'warhead': pattern,
                         'parsed_as': how,
                         'n_matches': len(matches),
                         'n_substituted': min(n_sub) if n_sub else None,
                         'intact_as_drawn': (min(n_sub) == 0) if n_sub else None,
                         'radical_in_match': any(d[1] for d in described),
                         'radical_adjacent': any(d[2] for d in described)})

    hits_df = pd.DataFrame(hits, columns=['mol_id', 'smiles', 'warhead_line',
                                          'warhead', 'parsed_as', 'n_matches',
                                          'n_substituted', 'intact_as_drawn',
                                          'radical_in_match', 'radical_adjacent'])
    hits_df = hits_df.sort_values(['mol_id', 'warhead_line'])
    hits_df.to_csv(f'{prefix}_warhead_hits.csv', index=False)
    print(f'wrote {prefix}_warhead_hits.csv')

    n_mols = len(molecules)
    counts = hits_df.groupby('warhead_line').agg(
        n_molecules=('mol_id', 'nunique'), n_matches=('n_matches', 'sum'))
    summary = pd.DataFrame([
        {'warhead_line': line_no,
         'warhead': pattern,
         'parsed_as': how,
         'n_molecules': int(counts.n_molecules.get(line_no, 0)),
         'fraction_of_molecules': counts.n_molecules.get(line_no, 0) / max(n_mols, 1),
         'n_matches': int(counts.n_matches.get(line_no, 0))}
        for line_no, pattern, query, how, reference in warheads])
    summary = summary.sort_values(['n_molecules', 'warhead_line'], ascending=[False, True])
    summary.to_csv(f'{prefix}_warhead_summary.csv', index=False)
    print(f'wrote {prefix}_warhead_summary.csv')

    n_with_hit = hits_df.mol_id.nunique()
    print()
    print(f'molecules containing at least one warhead: '
          f'{n_with_hit}/{n_mols} ({n_with_hit / max(n_mols, 1):.1%})')
    print(f'warheads present in at least one molecule:  '
          f'{(summary.n_molecules > 0).sum()}/{len(warheads)}')
    present = summary[summary.n_molecules > 0]
    if len(present):
        print()
        print(present[['warhead_line', 'warhead', 'parsed_as',
                       'n_molecules', 'n_matches']].to_string(index=False))


if __name__ == '__main__':
    main()
