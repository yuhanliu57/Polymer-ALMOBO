
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap
from polymetrix.featurizers.chemical_featurizer import (
    BalabanJIndex,
    BondCounts,
    BridgingRingsCount,
    FpDensityMorgan1,
    HalogenCounts,
    HeteroatomCount,
    HeteroatomDensity,
    MaxEStateIndex,
    MaxRingSize,
    MolecularWeight,
    NumAliphaticHeterocycles,
    NumAromaticRings,
    NumAtoms,
    NumHBondAcceptors,
    NumHBondDonors,
    NumNonAromaticRings,
    NumRings,
    NumRotatableBonds,
    SlogPVSA1,
    SmrVSA5,
    Sp2CarbonCountFeaturizer,
    Sp3CarbonCountFeaturizer,
    TopologicalSurfaceArea,
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.polymer import (
    Polymer,
    classify_backbone_and_sidechains,
    find_cycles_including_paths,
    find_shortest_paths_between_stars,
)
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    BackBoneFeaturizer,
    FullPolymerFeaturizer,
    NumBackBoneFeaturizer,
    NumSideChainFeaturizer,
    SideChainFeaturizer,
    SidechainDiversityFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, rdmolops
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

RDLogger.DisableLog("rdApp.*")

DERIVED_NAMES = [
    "TPSA_per_HA",
    "HBD_per_HA",
    "HBA_per_HA",
    "AroRing_per_HA",
    "Aro_to_SatRing_Ratio",
    "RotBonds_per_HA",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed=42):
    np.random.seed(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def read_measured_csvs(paths):
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}

        def pick(name):
            key = name.lower()
            if key not in cols:
                raise ValueError(f"Missing column '{name}' in {path}")
            return cols[key]

        df = df.rename(
            columns={
                pick("smiles"): "SMILES",
                pick("tc"): "TC",
                pick("modulus"): "Modulus",
            }
        )
        df = df[["SMILES", "TC", "Modulus"]].copy()
        df["SMILES"] = df["SMILES"].astype(str).str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid measured CSVs provided.")

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["SMILES", "TC", "Modulus"])
        .drop_duplicates(subset=["SMILES"], keep="last")
        .reset_index(drop=True)
    )


def non_dominated_mask(Y):
    n = Y.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        better = (Y >= Y[i]).all(axis=1)
        strict = (Y > Y[i]).any(axis=1)
        nd[i] = not np.any(better & strict)
    return nd


def compute_pf_idx(y_tc, y_mod):
    Y = MinMaxScaler().fit_transform(np.column_stack([y_tc, -y_mod]))
    return np.where(non_dominated_mask(Y))[0]


def _pm_build_featurizers():
    chem_core = [
        MolecularWeight(),
        NumRotatableBonds(),
        NumRings(),
        NumAromaticRings(),
        TopologicalSurfaceArea(),
        BalabanJIndex(),
        Sp2CarbonCountFeaturizer(),
        Sp3CarbonCountFeaturizer(),
        SmrVSA5(),
        SlogPVSA1(),
        FpDensityMorgan1(),
        HalogenCounts(),
        HeteroatomDensity(),
        BondCounts(),
        BridgingRingsCount(),
        MaxRingSize(),
        NumHBondDonors(),
        NumHBondAcceptors(),
        NumNonAromaticRings(),
        NumAliphaticHeterocycles(),
        MaxEStateIndex(),
    ]
    full = [FullPolymerFeaturizer(f) for f in chem_core]
    backbone = [
        NumBackBoneFeaturizer(),
        BackBoneFeaturizer(NumAtoms()),
        BackBoneFeaturizer(NumAromaticRings()),
        BackBoneFeaturizer(NumRotatableBonds()),
        BackBoneFeaturizer(TopologicalSurfaceArea()),
        BackBoneFeaturizer(BalabanJIndex()),
    ]
    sidechain = [
        NumSideChainFeaturizer(),
        SideChainFeaturizer(NumAtoms(agg=["sum"])),
        SideChainFeaturizer(NumRotatableBonds(agg=["sum"])),
        SideChainFeaturizer(NumAromaticRings(agg=["sum"])),
        SideChainFeaturizer(TopologicalSurfaceArea(agg=["sum"])),
        SidechainLengthToStarAttachmentDistanceRatioFeaturizer(),
        StarToSidechainMinDistanceFeaturizer(),
        SidechainDiversityFeaturizer(),
    ]
    return MultipleFeaturizer(full), MultipleFeaturizer(backbone), MultipleFeaturizer(sidechain)


def _prefix_labels(labels, prefix):
    return [f"{prefix} {label}" for label in labels]


def compute_polymetrix_features(smiles_list):
    full_multi, backbone_multi, sidechain_multi = _pm_build_featurizers()
    rows = []
    labels = None

    for smiles in smiles_list:
        polymer = Polymer.from_psmiles(smiles)
        x_full = full_multi.featurize(polymer)
        x_backbone = backbone_multi.featurize(polymer)
        x_sidechain = sidechain_multi.featurize(polymer)

        if labels is None:
            labels = (
                _prefix_labels(full_multi.feature_labels(), "[F]")
                + _prefix_labels(backbone_multi.feature_labels(), "[B]")
                + _prefix_labels(sidechain_multi.feature_labels(), "[S]")
            )

        rows.append(np.hstack([x_full, x_backbone, x_sidechain]))

    return np.asarray(rows, dtype=float), labels


def _classify_nodes_with_pm(psmiles):
    polymer = Polymer.from_psmiles(psmiles)
    try:
        backbone_nodes, sidechain_nodes = classify_backbone_and_sidechains(polymer.graph)
        return list(backbone_nodes), list(sidechain_nodes)
    except TypeError:
        try:
            paths = find_shortest_paths_between_stars(polymer.graph)
            find_cycles_including_paths(polymer.graph, paths)
            backbone_nodes, sidechain_nodes = classify_backbone_and_sidechains(polymer.graph, paths)
            return list(backbone_nodes), list(sidechain_nodes)
        except Exception:
            return [], []
    except Exception:
        return [], []


def _submol_keep_atoms(mol, keep):
    keep = set(keep)
    editable = Chem.RWMol(mol)
    remove = [i for i in range(mol.GetNumAtoms()) if i not in keep]
    for idx in sorted(remove, reverse=True):
        editable.RemoveAtom(idx)
    submol = editable.GetMol()
    try:
        Chem.SanitizeMol(submol)
    except Exception:
        try:
            submol = Chem.MolFromSmiles(Chem.MolToSmiles(submol))
        except Exception:
            pass
    return submol


def _rdkit_block(mol):
    fields = [
        "MolLogP",
        "MolMR",
        "LabuteASA",
        "MaxAbsQ",
        "MinAbsQ",
        "HeavyAtomCount",
        "BertzCT",
        "Kappa1",
        "Kappa2",
        "FractionCSP3",
        "NumConjBonds",
        "ConjBondFrac",
        "AromaticAtomFrac",
        "nO",
        "nN",
        "nHal",
        "HalogenFrac",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "LongestPathLen",
        "fr_ether",
        "fr_ester",
        "fr_amide",
    ]
    if mol is None:
        return {field: 0.0 for field in fields}

    def safe(value, default=0.0):
        try:
            value = float(value)
            return value if np.isfinite(value) else default
        except Exception:
            return default

    heavy_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    num_bonds = mol.GetNumBonds()
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    num_conjugated_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsConjugated())
    num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    num_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    num_halogen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (9, 17, 35, 53))

    longest_path = 0.0
    try:
        distance = rdmolops.GetDistanceMatrix(mol)
        heavy_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        if len(heavy_idx) >= 2:
            longest_path = float(np.max(distance[np.ix_(heavy_idx, heavy_idx)]))
    except Exception:
        longest_path = 0.0

    max_abs_q = 0.0
    min_abs_q = 0.0
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                if np.isfinite(charge):
                    charges.append(abs(charge))
        if charges:
            max_abs_q = max(charges)
            min_abs_q = min(charges)
    except Exception:
        pass

    aromatic_frac = num_aromatic_atoms / heavy_atom_count if heavy_atom_count else 0.0
    conj_frac = num_conjugated_bonds / num_bonds if num_bonds else 0.0
    halogen_frac = num_halogen / heavy_atom_count if heavy_atom_count else 0.0

    return {
        "MolLogP": safe(Descriptors.MolLogP(mol)),
        "MolMR": safe(Descriptors.MolMR(mol)),
        "LabuteASA": safe(rdMolDescriptors.CalcLabuteASA(mol)),
        "MaxAbsQ": max_abs_q,
        "MinAbsQ": min_abs_q,
        "HeavyAtomCount": float(heavy_atom_count),
        "BertzCT": safe(Descriptors.BertzCT(mol)),
        "Kappa1": safe(Descriptors.Kappa1(mol)),
        "Kappa2": safe(Descriptors.Kappa2(mol)),
        "FractionCSP3": safe(Descriptors.FractionCSP3(mol)),
        "NumConjBonds": float(num_conjugated_bonds),
        "ConjBondFrac": float(conj_frac),
        "AromaticAtomFrac": float(aromatic_frac),
        "nO": float(num_oxygen),
        "nN": float(num_nitrogen),
        "nHal": float(num_halogen),
        "HalogenFrac": float(halogen_frac),
        "NumAliphaticRings": safe(Descriptors.NumAliphaticRings(mol)),
        "NumSaturatedRings": safe(Descriptors.NumSaturatedRings(mol)),
        "LongestPathLen": float(longest_path),
        "fr_ether": float(Fragments.fr_ether(mol)),
        "fr_ester": float(Fragments.fr_ester(mol)),
        "fr_amide": safe(rdMolDescriptors.CalcNumAmideBonds(mol), float(Fragments.fr_amide(mol))),
    }


def compute_rdkit_supplement(smiles_list):
    props = [
        "MolLogP",
        "MolMR",
        "LabuteASA",
        "MaxAbsQ",
        "MinAbsQ",
        "HeavyAtomCount",
        "BertzCT",
        "Kappa1",
        "Kappa2",
        "FractionCSP3",
        "NumConjBonds",
        "ConjBondFrac",
        "AromaticAtomFrac",
        "nO",
        "nN",
        "nHal",
        "HalogenFrac",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "LongestPathLen",
        "fr_ether",
        "fr_ester",
        "fr_amide",
    ]
    labels = [f"[F] {name}" for name in props] + [f"[B] {name}" for name in props] + [f"[S] {name}" for name in props]

    rows = []
    for smiles in smiles_list:
        mol_full = Chem.MolFromSmiles(smiles)
        if mol_full is None:
            rows.append(np.zeros(len(labels), dtype=float))
            continue

        backbone_nodes, sidechain_nodes = _classify_nodes_with_pm(smiles)
        mol_backbone = _submol_keep_atoms(mol_full, backbone_nodes) if backbone_nodes else Chem.MolFromSmiles("*")
        mol_sidechain = _submol_keep_atoms(mol_full, sidechain_nodes) if sidechain_nodes else Chem.MolFromSmiles("*")

        desc_full = _rdkit_block(mol_full)
        desc_backbone = _rdkit_block(mol_backbone)
        desc_sidechain = _rdkit_block(mol_sidechain)

        rows.append(
            np.asarray(
                [desc_full[name] for name in props]
                + [desc_backbone[name] for name in props]
                + [desc_sidechain[name] for name in props],
                dtype=float,
            )
        )

    return np.vstack(rows), labels


def append_derived_features_slim(X, names):
    name_to_idx = {name: i for i, name in enumerate(names)}

    def get(keys):
        for key in keys:
            if key in name_to_idx:
                return X[:, name_to_idx[key]]
        return np.zeros(X.shape[0], dtype=float)

    tpsa = get(["[F] topological_surface_area_sum_fullpolymerfeaturizer", "[F] TopologicalSurfaceArea", "TPSA"])
    hbd = get(["[F] num_hbond_donors_sum_fullpolymerfeaturizer", "NumHDonors"])
    hba = get(["[F] num_hbond_acceptors_sum_fullpolymerfeaturizer", "NumHAcceptors"])
    aro = get(["[F] num_aromatic_rings_sum_fullpolymerfeaturizer", "NumAromaticRings"])
    rot = get(["[F] num_rotatable_bonds_sum_fullpolymerfeaturizer", "NumRotatableBonds"])
    sat = get(["[F] NumSaturatedRings", "[F] num_non_aromatic_rings_sum_fullpolymerfeaturizer"])
    heavy = np.maximum(get(["[F] HeavyAtomCount"]), 1.0)

    eps = 1e-8
    derived = np.column_stack(
        [
            tpsa / heavy,
            hbd / heavy,
            hba / heavy,
            aro / heavy,
            (aro + eps) / (sat + eps),
            rot / heavy,
        ]
    )
    return np.hstack([X, derived]), names + DERIVED_NAMES


def sanitize_and_scale(X_raw):
    X_raw = np.asarray(X_raw, dtype=float)
    X_raw[~np.isfinite(X_raw)] = np.nan

    medians = np.nanmedian(X_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    rows, cols = np.where(~np.isfinite(X_raw))
    if rows.size:
        X_raw[rows, cols] = medians[cols]

    np.clip(X_raw, -1e6, 1e6, out=X_raw)
    scaler = StandardScaler().fit(X_raw)
    return scaler.transform(X_raw)


def drop_correlated_features(X_std, names, threshold=0.92, order_by=None):
    corr = np.corrcoef(X_std, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    scores = np.nan_to_num(order_by) if order_by is not None else np.nan_to_num(np.std(X_std, axis=0))
    order = np.argsort(-scores)

    keep = set()
    banned = set()
    for j in order:
        if j in banned:
            continue
        keep.add(j)
        neighbors = np.where(np.abs(corr[j]) > threshold)[0]
        for k in neighbors:
            if k != j:
                banned.add(k)

    keep_idx = sorted(keep)
    return X_std[:, keep_idx], [names[i] for i in keep_idx], keep_idx


def scaffold_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or ""
    except Exception:
        return ""


def build_tree_model(alg, params):
    model_cls = RandomForestRegressor if alg == "rf" else ExtraTreesRegressor
    return model_cls(**params)


def tune_tree_model(X, y, scaffolds, seed=42, do_tune=True, tune_mode="kfold"):
    base_params = {
        "n_estimators": 1200,
        "min_samples_leaf": 6,
        "max_features": 0.5,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": seed,
    }
    if not do_tune:
        return "rf", base_params

    if tune_mode == "scaffold" and len(np.unique(scaffolds)) >= 2:
        splitter = GroupKFold(n_splits=min(7, max(3, len(np.unique(scaffolds)))))

        def folds():
            yield from splitter.split(X, y, groups=scaffolds)

    else:
        splitter = KFold(n_splits=10, shuffle=True, random_state=seed)

        def folds():
            yield from splitter.split(X, y)

    grid = []
    for alg in ("rf", "et"):
        for n_estimators in (800, 1200, 1600):
            for min_samples_leaf in (4, 6, 10):
                for max_features in (0.5, 0.7, "sqrt"):
                    grid.append(
                        (
                            alg,
                            {
                                "n_estimators": n_estimators,
                                "min_samples_leaf": min_samples_leaf,
                                "max_features": max_features,
                                "max_depth": None,
                                "bootstrap": alg == "rf",
                                "n_jobs": -1,
                                "random_state": seed,
                            },
                        )
                    )

    best_alg = "rf"
    best_params = base_params
    best_score = -np.inf

    for alg, params in grid:
        scores = []
        for train_idx, test_idx in folds():
            model = build_tree_model(alg, params).fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            scores.append(r2_score(y[test_idx], pred))
        score = float(np.mean(scores))
        if score > best_score:
            best_alg = alg
            best_params = params
            best_score = score

    return best_alg, best_params


def evaluate_random_split(X, y_tc, y_mod, y_mod_trans, alg_tc, params_tc, alg_mod, params_mod, inv_mod, seed):
    idx_all = np.arange(X.shape[0])

    X_train_tc, X_test_tc, y_train_tc, y_test_tc = train_test_split(X, y_tc, test_size=0.2, random_state=seed)
    model_tc = build_tree_model(alg_tc, params_tc).fit(X_train_tc, y_train_tc)
    pred_tc = model_tc.predict(X_test_tc)

    X_train_mod, X_test_mod, y_train_mod, y_test_mod, idx_train, idx_test = train_test_split(
        X, y_mod_trans, idx_all, test_size=0.2, random_state=seed
    )
    model_mod = build_tree_model(alg_mod, params_mod).fit(X_train_mod, y_train_mod)
    pred_mod = inv_mod(model_mod.predict(X_test_mod))

    return {
        "tc_r2": float(r2_score(y_test_tc, pred_tc)),
        "tc_rmse": rmse(y_test_tc, pred_tc),
        "mod_r2": float(r2_score(y_mod[idx_test], pred_mod)),
        "mod_rmse": rmse(y_mod[idx_test], pred_mod),
    }


def evaluate_repeated_kfold(X, y_tc, y_mod, y_mod_trans, alg_tc, params_tc, alg_mod, params_mod, inv_mod, n_splits, n_repeats, seed):
    tc_r2, tc_rmse = [], []
    mod_r2, mod_rmse = [], []

    splitter_tc = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for train_idx, test_idx in splitter_tc.split(X, y_tc):
        pred_tc = build_tree_model(alg_tc, params_tc).fit(X[train_idx], y_tc[train_idx]).predict(X[test_idx])
        tc_r2.append(r2_score(y_tc[test_idx], pred_tc))
        tc_rmse.append(rmse(y_tc[test_idx], pred_tc))

    splitter_mod = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed + 1)
    for train_idx, test_idx in splitter_mod.split(X, y_mod_trans):
        pred_mod = build_tree_model(alg_mod, params_mod).fit(X[train_idx], y_mod_trans[train_idx]).predict(X[test_idx])
        pred_mod = inv_mod(pred_mod)
        mod_r2.append(r2_score(y_mod[test_idx], pred_mod))
        mod_rmse.append(rmse(y_mod[test_idx], pred_mod))

    return {
        "tc_r2_mean": float(np.mean(tc_r2)),
        "tc_r2_std": float(np.std(tc_r2)),
        "tc_rmse_mean": float(np.mean(tc_rmse)),
        "tc_rmse_std": float(np.std(tc_rmse)),
        "mod_r2_mean": float(np.mean(mod_r2)),
        "mod_r2_std": float(np.std(mod_r2)),
        "mod_rmse_mean": float(np.mean(mod_rmse)),
        "mod_rmse_std": float(np.std(mod_rmse)),
    }


def evaluate_scaffold_kfold(X, y_tc, y_mod, y_mod_trans, scaffolds, alg_tc, params_tc, alg_mod, params_mod, inv_mod):
    unique_scaffolds = np.unique(scaffolds)
    if len(unique_scaffolds) < 2:
        return None

    tc_r2, tc_rmse = [], []
    mod_r2, mod_rmse = [], []

    splitter_tc = GroupKFold(n_splits=min(5, len(unique_scaffolds)))
    for train_idx, test_idx in splitter_tc.split(X, y_tc, groups=scaffolds):
        pred_tc = build_tree_model(alg_tc, params_tc).fit(X[train_idx], y_tc[train_idx]).predict(X[test_idx])
        tc_r2.append(r2_score(y_tc[test_idx], pred_tc))
        tc_rmse.append(rmse(y_tc[test_idx], pred_tc))

    splitter_mod = GroupKFold(n_splits=min(5, len(unique_scaffolds)))
    for train_idx, test_idx in splitter_mod.split(X, y_mod_trans, groups=scaffolds):
        pred_mod = build_tree_model(alg_mod, params_mod).fit(X[train_idx], y_mod_trans[train_idx]).predict(X[test_idx])
        pred_mod = inv_mod(pred_mod)
        mod_r2.append(r2_score(y_mod[test_idx], pred_mod))
        mod_rmse.append(rmse(y_mod[test_idx], pred_mod))

    return {
        "tc_r2_mean": float(np.mean(tc_r2)),
        "tc_r2_std": float(np.std(tc_r2)),
        "tc_rmse_mean": float(np.mean(tc_rmse)),
        "tc_rmse_std": float(np.std(tc_rmse)),
        "mod_r2_mean": float(np.mean(mod_r2)),
        "mod_r2_std": float(np.std(mod_r2)),
        "mod_rmse_mean": float(np.mean(mod_rmse)),
        "mod_rmse_std": float(np.std(mod_rmse)),
    }


def write_metric_reports(results_dir, random_metrics, kfold_metrics, scaffold_metrics, n_splits, n_repeats):
    with open(results_dir / "metrics_random.txt", "w", encoding="utf-8") as f:
        f.write(f"Random split 80/20 — TC:  R2={random_metrics['tc_r2']:.4f}, RMSE={random_metrics['tc_rmse']:.5f}\n")
        f.write(f"Random split 80/20 — Modulus: R2={random_metrics['mod_r2']:.4f}, RMSE={random_metrics['mod_rmse']:.5f}\n")

    with open(results_dir / "metrics_kfold.txt", "w", encoding="utf-8") as f:
        f.write(f"Repeated KFold (n_splits={n_splits}, n_repeats={n_repeats}) — mean±sd\n")
        f.write(
            f"TC : R2={kfold_metrics['tc_r2_mean']:.4f}±{kfold_metrics['tc_r2_std']:.4f}, "
            f"RMSE={kfold_metrics['tc_rmse_mean']:.5f}±{kfold_metrics['tc_rmse_std']:.5f}\n"
        )
        f.write(
            f"Modulus: R2={kfold_metrics['mod_r2_mean']:.4f}±{kfold_metrics['mod_r2_std']:.4f}, "
            f"RMSE={kfold_metrics['mod_rmse_mean']:.5f}±{kfold_metrics['mod_rmse_std']:.5f}\n"
        )

    if scaffold_metrics is not None:
        with open(results_dir / "metrics_scaffold.txt", "w", encoding="utf-8") as f:
            f.write("Scaffold GroupKFold — mean±sd\n")
            f.write(
                f"TC : R2={scaffold_metrics['tc_r2_mean']:.4f}±{scaffold_metrics['tc_r2_std']:.4f}, "
                f"RMSE={scaffold_metrics['tc_rmse_mean']:.5f}±{scaffold_metrics['tc_rmse_std']:.5f}\n"
            )
            f.write(
                f"Modulus: R2={scaffold_metrics['mod_r2_mean']:.4f}±{scaffold_metrics['mod_r2_std']:.4f}, "
                f"RMSE={scaffold_metrics['mod_rmse_mean']:.5f}±{scaffold_metrics['mod_rmse_std']:.5f}\n"
            )


def oof_shap_values(X, y, alg, params, seed=42, k_shap_folds=10):
    values = np.zeros_like(X, dtype=float)
    splitter = KFold(n_splits=k_shap_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        model = build_tree_model(alg, params).fit(X[train_idx], y[train_idx])

        rng = np.random.RandomState(seed + fold)
        bg_idx = rng.choice(train_idx, size=min(500, len(train_idx)), replace=False)
        background = X[bg_idx]

        try:
            explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        except Exception:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X[test_idx], check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        values[test_idx] = shap_values

    return values


def full_shap_values(model, X, seed=0):
    rng = np.random.RandomState(seed)
    bg_idx = rng.choice(np.arange(X.shape[0]), size=min(500, X.shape[0]), replace=False)
    background = X[bg_idx]

    try:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    except Exception:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)
    return shap_values[0] if isinstance(shap_values, list) else shap_values


def sign_consistency_on_pf(shap_values, pf_idx):
    subset = shap_values[pf_idx]
    positive = np.mean(subset >= 0, axis=0)
    negative = 1.0 - positive
    return np.maximum(positive, negative)


def bootstrap_topk_stability(shap_values, names, topk, n_boot, seed=42):
    rng = np.random.RandomState(seed)
    n = shap_values.shape[0]
    base = [names[i] for i in np.argsort(-np.mean(np.abs(shap_values), axis=0))[:topk]]

    freq = pd.Series(0.0, index=names)
    jaccard = []

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        sample_topk = [names[i] for i in np.argsort(-np.mean(np.abs(shap_values[idx]), axis=0))[:topk]]
        freq[sample_topk] += 1.0
        inter = len(set(base).intersection(sample_topk))
        union = len(set(base).union(sample_topk))
        jaccard.append(inter / union if union else 0.0)

    return base, (freq / n_boot).sort_values(ascending=False), float(np.mean(jaccard)), float(np.std(jaccard))


def y_scramble_overlap(X, y, names, shap_values, alg, params, topk, n_times, seed=42, k_shap_folds=10):
    rng = np.random.RandomState(seed)
    base_topk = [names[i] for i in np.argsort(-np.mean(np.abs(shap_values), axis=0))[:topk]]

    overlaps = []
    jaccard = []
    for i in range(n_times):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        perm_values = oof_shap_values(X, y_perm, alg, params, seed=seed + i + 1, k_shap_folds=k_shap_folds)
        perm_topk = [names[j] for j in np.argsort(-np.mean(np.abs(perm_values), axis=0))[:topk]]
        inter = len(set(base_topk).intersection(perm_topk))
        union = len(set(base_topk).union(perm_topk))
        overlaps.append(inter)
        jaccard.append(inter / union if union else 0.0)

    return base_topk, float(np.mean(overlaps)), float(np.std(overlaps)), float(np.mean(jaccard)), float(np.std(jaccard))


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_compute_shap(
    init_csv,
    selected_csv,
    results_dir,
    random_state=42,
    topk=15,
    winsor_pct=2.5,
    target_transform="yeo-johnson",
    do_tune=True,
    tune_mode="kfold",
    k_shap_folds=10,
    k_cv_folds=10,
    k_cv_repeats=3,
    boot_n=500,
    yscr_n=20,
    cons_thr=0.70,
):
    set_seed(random_state)
    results_dir = ensure_dir(results_dir)

    measured = read_measured_csvs([init_csv, selected_csv])
    smiles = measured["SMILES"].astype(str).tolist()
    y_tc = measured["TC"].to_numpy(dtype=float)
    y_mod = measured["Modulus"].to_numpy(dtype=float)

    if winsor_pct and winsor_pct > 0:
        lo, hi = np.percentile(y_mod, [winsor_pct, 100.0 - winsor_pct])
        y_mod = np.clip(y_mod, lo, hi)

    if target_transform == "yeo-johnson":
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        y_mod_train = transformer.fit_transform(y_mod.reshape(-1, 1)).ravel()
        inv_mod = lambda z: transformer.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    elif target_transform == "log1p":
        y_mod_train = np.log1p(y_mod)
        inv_mod = lambda z: np.expm1(z)
    else:
        y_mod_train = y_mod.copy()
        inv_mod = lambda z: np.asarray(z)

    X_pm, names_pm = compute_polymetrix_features(smiles)
    X_rd, names_rd = compute_rdkit_supplement(smiles)
    X_raw = np.hstack([X_pm, X_rd])
    names_all = names_pm + names_rd
    X_raw, names_all = append_derived_features_slim(X_raw, names_all)

    raw_std = np.nanstd(X_raw, axis=0)
    X_std = sanitize_and_scale(X_raw)
    X_kept, names_kept, keep_idx = drop_correlated_features(X_std, names_all, threshold=0.92, order_by=raw_std)

    pd.DataFrame({"Descriptor": names_kept}).to_csv(results_dir / "descriptors_kept.csv", index=False)

    scaffolds = np.asarray([scaffold_smiles(smiles_i) for smiles_i in smiles])

    alg_tc, params_tc = tune_tree_model(X_kept, y_tc, scaffolds, seed=random_state, do_tune=do_tune, tune_mode=tune_mode)
    alg_mod, params_mod = tune_tree_model(X_kept, y_mod_train, scaffolds, seed=random_state, do_tune=do_tune, tune_mode=tune_mode)

    model_tc = build_tree_model(alg_tc, params_tc).fit(X_kept, y_tc)
    model_mod = build_tree_model(alg_mod, params_mod).fit(X_kept, y_mod_train)

    random_metrics = evaluate_random_split(
        X_kept,
        y_tc,
        y_mod,
        y_mod_train,
        alg_tc,
        params_tc,
        alg_mod,
        params_mod,
        inv_mod,
        random_state,
    )
    kfold_metrics = evaluate_repeated_kfold(
        X_kept,
        y_tc,
        y_mod,
        y_mod_train,
        alg_tc,
        params_tc,
        alg_mod,
        params_mod,
        inv_mod,
        k_cv_folds,
        k_cv_repeats,
        random_state,
    )
    scaffold_metrics = evaluate_scaffold_kfold(
        X_kept,
        y_tc,
        y_mod,
        y_mod_train,
        scaffolds,
        alg_tc,
        params_tc,
        alg_mod,
        params_mod,
        inv_mod,
    )
    write_metric_reports(results_dir, random_metrics, kfold_metrics, scaffold_metrics, k_cv_folds, k_cv_repeats)

    shap_tc_oof = oof_shap_values(X_kept, y_tc, alg_tc, params_tc, seed=random_state, k_shap_folds=k_shap_folds)
    shap_mod_oof = oof_shap_values(X_kept, y_mod_train, alg_mod, params_mod, seed=random_state + 11, k_shap_folds=k_shap_folds)
    shap_tc_full = full_shap_values(model_tc, X_kept, seed=0)
    shap_mod_full = full_shap_values(model_mod, X_kept, seed=1)

    pf_idx = compute_pf_idx(y_tc, y_mod)
    cons_tc = sign_consistency_on_pf(shap_tc_oof, pf_idx)
    cons_mod = sign_consistency_on_pf(shap_mod_oof, pf_idx)

    top_tc, freq_tc, jaccard_tc_mean, jaccard_tc_std = bootstrap_topk_stability(
        shap_tc_oof,
        names_kept,
        topk,
        boot_n,
        seed=random_state + 7,
    )
    top_mod, freq_mod, jaccard_mod_mean, jaccard_mod_std = bootstrap_topk_stability(
        shap_mod_oof,
        names_kept,
        topk,
        boot_n,
        seed=random_state + 13,
    )
    freq_tc.to_csv(results_dir / "stability_topk_TC.csv", header=["freq_in_topK"])
    freq_mod.to_csv(results_dir / "stability_topk_MOD.csv", header=["freq_in_topK"])

    base_tc_y, overlap_tc_mean, overlap_tc_std, yscramble_tc_mean, yscramble_tc_std = y_scramble_overlap(
        X_kept,
        y_tc,
        names_kept,
        shap_tc_oof,
        alg_tc,
        params_tc,
        topk,
        yscr_n,
        seed=random_state + 21,
        k_shap_folds=k_shap_folds,
    )
    base_mod_y, overlap_mod_mean, overlap_mod_std, yscramble_mod_mean, yscramble_mod_std = y_scramble_overlap(
        X_kept,
        y_mod_train,
        names_kept,
        shap_mod_oof,
        alg_mod,
        params_mod,
        topk,
        yscr_n,
        seed=random_state + 31,
        k_shap_folds=k_shap_folds,
    )

    with open(results_dir / "stability_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"TC  Top-{topk} base: {top_tc}\n")
        f.write(f"TC  Jaccard(mean±sd) over {boot_n} bootstraps: {jaccard_tc_mean:.3f}±{jaccard_tc_std:.3f}\n")
        f.write(f"Modulus Top-{topk} base: {top_mod}\n")
        f.write(f"MOD Jaccard(mean±sd) over {boot_n} bootstraps: {jaccard_mod_mean:.3f}±{jaccard_mod_std:.3f}\n")

    with open(results_dir / "yscramble_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"TC  base Top-{topk}: {base_tc_y}\n")
        f.write(
            f"TC  y-scramble (n={yscr_n}) overlap mean±sd: {overlap_tc_mean:.2f}±{overlap_tc_std:.2f}; "
            f"Jaccard {yscramble_tc_mean:.3f}±{yscramble_tc_std:.3f}\n"
        )
        f.write(f"Modulus base Top-{topk}: {base_mod_y}\n")
        f.write(
            f"MOD y-scramble (n={yscr_n}) overlap mean±sd: {overlap_mod_mean:.2f}±{overlap_mod_std:.2f}; "
            f"Jaccard {yscramble_mod_mean:.3f}±{yscramble_mod_std:.3f}\n"
        )

    analysis_stats = {
        "random_state": random_state,
        "topk": topk,
        "cons_thr": cons_thr,
        "target_transform": target_transform or "None",
        "bootstrap": {
            "tc_jaccard_mean": jaccard_tc_mean,
            "tc_jaccard_std": jaccard_tc_std,
            "mod_jaccard_mean": jaccard_mod_mean,
            "mod_jaccard_std": jaccard_mod_std,
        },
        "yscramble": {
            "tc_overlap_mean": overlap_tc_mean,
            "tc_overlap_std": overlap_tc_std,
            "tc_jaccard_mean": yscramble_tc_mean,
            "tc_jaccard_std": yscramble_tc_std,
            "mod_overlap_mean": overlap_mod_mean,
            "mod_overlap_std": overlap_mod_std,
            "mod_jaccard_mean": yscramble_mod_mean,
            "mod_jaccard_std": yscramble_mod_std,
        },
        "consistency_counts": {
            "tc_ge_thr": int(np.sum(cons_tc >= cons_thr)),
            "tc_total": int(len(cons_tc)),
            "mod_ge_thr": int(np.sum(cons_mod >= cons_thr)),
            "mod_total": int(len(cons_mod)),
        },
    }
    save_json(results_dir / "analysis_stats.json", analysis_stats)

    model_metadata = {
        "alg_tc": alg_tc,
        "alg_mod": alg_mod,
        "params_tc": params_tc,
        "params_mod": params_mod,
        "target_transform": target_transform or "None",
        "topk": topk,
        "keep_idx": keep_idx,
    }
    save_json(results_dir / "model_metadata.json", model_metadata)

    np.save(results_dir / "X_kept.npy", X_kept)
    np.save(results_dir / "y_tc.npy", y_tc)
    np.save(results_dir / "y_mod.npy", y_mod)
    np.save(results_dir / "pf_idx.npy", pf_idx)
    np.save(results_dir / "cons_tc.npy", cons_tc)
    np.save(results_dir / "cons_mod.npy", cons_mod)
    np.save(results_dir / "shap_oof_TC.npy", shap_tc_oof)
    np.save(results_dir / "shap_oof_MOD.npy", shap_mod_oof)
    np.save(results_dir / "shap_full_TC.npy", shap_tc_full)
    np.save(results_dir / "shap_full_MOD.npy", shap_mod_full)

    save_json(results_dir / "names_kept.json", names_kept)

    return {
        "results_dir": str(results_dir),
        "alg_tc": alg_tc,
        "alg_mod": alg_mod,
        "params_tc": params_tc,
        "params_mod": params_mod,
        "num_descriptors": int(len(names_kept)),
        "num_samples": int(len(smiles)),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]

    init_csv = project_root / "data" / "Initial_set.csv"
    selected_csv = project_root / "results" / "all_candidates.csv"
    results_dir = project_root / "shap_postanalysis" / "results"

    run_compute_shap(
        init_csv=init_csv,
        selected_csv=selected_csv,
        results_dir=results_dir,
        random_state=42,
        topk=15,
        winsor_pct=2.5,
        target_transform="yeo-johnson",
        do_tune=True,
        tune_mode="kfold",
        k_shap_folds=10,
        k_cv_folds=10,
        k_cv_repeats=3,
        boot_n=500,
        yscr_n=20,
        cons_thr=0.70,
    )


if __name__ == "__main__":
    main()
