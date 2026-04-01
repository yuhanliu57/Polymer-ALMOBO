import warnings

import numpy as np
import pandas as pd
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from rdkit import Chem, RDLogger
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")


def _read_csv(csv_path, required_columns):
    df = pd.read_csv(csv_path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {missing}")
    return df


def smiles_to_embeddings(smiles_list, model, unseen="UNK"):
    kv = model.wv
    zero_vec = np.zeros(kv.vector_size, dtype=float)
    embeddings = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.warn(f"Invalid SMILES: {smi}, using zero vector")
            embeddings.append(zero_vec.copy())
            continue

        tokens = mol2alt_sentence(mol, 1)
        vecs = [
            kv[token]
            if token in kv.key_to_index
            else (kv[unseen] if unseen in kv.key_to_index else zero_vec)
            for token in tokens
        ]
        embeddings.append(np.mean(vecs, axis=0) if vecs else zero_vec)

    return np.asarray(embeddings, dtype=float)


def prepare_data_train(csv_path, w2v_model):
    df = _read_csv(csv_path, {"SMILES", "TC", "Modulus"})
    X_raw = smiles_to_embeddings(df["SMILES"].values, w2v_model)
    scaler_X = StandardScaler().fit(X_raw)
    return scaler_X.transform(X_raw), df, scaler_X


def load_unlabeled(csv_path, w2v_model, scaler_X):
    df = _read_csv(csv_path, {"PID", "SMILES"})
    X_raw = smiles_to_embeddings(df["SMILES"].values, w2v_model)
    return scaler_X.transform(X_raw), df


def load_word2vec(w2v_path):
    return word2vec.Word2Vec.load(w2v_path)
