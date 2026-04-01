
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Rectangle

_ALIAS_PATTERNS = [
    (r"\btopological[_\s]+surface[_\s]+area\b", "TPSA"),
    (r"\btpsa\b", "TPSA"),
    (r"\bbalaban[_\s]*j(?:[_\s]*index)?\b", "BalabanJ"),
    (r"\bnum[_\s]+rotatable[_\s]+bonds\b", "RotBonds"),
    (r"\brotbonds\b", "RotBonds"),
    (r"\bfractioncsp3\b", "FracCSP3"),
    (r"\baromaticatomfrac\b", "AromaticFrac"),
    (r"\bnum[_\s]+aromatic[_\s]+rings\b", "AroRings"),
    (r"\bmollogp\b", "MolLogP"),
    (r"\bmaxabsq\b", "MaxAbsQ"),
    (r"\bhalogenfrac\b", "HalogenFrac"),
    (r"\bconjbondfrac\b", "ConjBondFrac"),
    (r"\bnumconjbonds\b", "NumConjBonds"),
    (r"\bnumaliphaticrings\b", "AliphRings"),
    (r"\bnumsaturatedrings\b", "SatRings"),
    (r"\blabuteasa\b", "LabuteASA"),
    (r"\bnum[_\s]+diverse[_\s]+sidechains\b", "SidechainDiv"),
    (r"\bstar[_\s]+to[_\s]+sidechain[_\s]+min[_\s]+distance\b", "Star-SC minDist"),
    (r"\bminabsq\b", "MinAbsQ"),
    (r"\bnumsidechainfeaturizer\b", "NumSidechains"),
    (r"\bfr[_\s]+ether\b", "fr-ether"),
    (r"\bfr[_\s]+ester\b", "fr-ester"),
    (r"\bfr[_\s]+amide\b", "fr-amide"),
    (r"\bno\b", "nO"),
    (r"\bnn\b", "nN"),
    (r"\bnhal\b", "nHal"),
    (r"\bha\b", "HA"),
    (r"\bsmr[_\s]*vsa5\b", "SmrVSA5"),
    (r"\bslogp[_\s]*vsa1\b", "SlogPVSA1"),
    (r"\bfp[_\s]*density[_\s]*morgan1\b", "FpDensityMorgan1"),
    (r"\bmax[_\s]*estate[_\s]*index\b", "MaxEState"),
    (r"\bmolecular[_\s]*weight\b", "MolWt"),
    (r"\bnum[_\s]*hbond[_\s]*donors\b", "HBD"),
    (r"\bnum[_\s]*hbond[_\s]*acceptors\b", "HBA"),
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _strip_suffixes(name):
    name = re.sub(r"_(fullpolymerfeaturizer|backbonefeaturizer|sidechainfeaturizer|featurizer)(_sum)?\b", "", name, flags=re.I)
    name = re.sub(r"_sum\b", "", name, flags=re.I)
    return name


def _short_label(name, max_chars=28):
    match = re.match(r"^\s*\[([FBSfbs])\]\s*", name)
    layer = f" ({match.group(1).upper()})" if match else ""

    label = re.sub(r"^\s*\[[FBSfbs]\]\s*", "", name)
    label = _strip_suffixes(label.lower())
    label = re.sub(r"[_\s]+", " ", label).strip()

    for pattern, replacement in _ALIAS_PATTERNS:
        label = re.sub(pattern, replacement, label, flags=re.I)

    keep_tokens = {
        "TPSA",
        "RotBonds",
        "FracCSP3",
        "AromaticFrac",
        "AroRings",
        "MolLogP",
        "MaxAbsQ",
        "MinAbsQ",
        "ConjBondFrac",
        "NumConjBonds",
        "AliphRings",
        "SatRings",
        "BalabanJ",
        "SidechainDiv",
        "NumSidechains",
        "Star-SC",
        "minDist",
        "fr-ether",
        "fr-ester",
        "fr-amide",
        "nO",
        "nN",
        "nHal",
        "HA",
    }
    lower_tokens = {"per", "to", "of", "and"}

    words = []
    for word in label.split():
        if word in keep_tokens or "-" in word or "/" in word or not word.islower():
            words.append(word)
        elif word in lower_tokens:
            words.append(word)
        else:
            words.append(word.capitalize())

    out = " ".join(words) + layer
    return out if len(out) <= max_chars else out[: max_chars - 1] + "..."


def _abbreviate_names(names):
    return [_short_label(name) for name in names]


def plot_beeswarm_filtered(shap_values, X_std, feature_names, out_png, topk=15, include_prefixes=("[B]", "[S]"), xlabel=None, xticks_step=None):
    idx = [i for i, name in enumerate(feature_names) if any(name.startswith(prefix) for prefix in include_prefixes)]
    if not idx:
        return

    shap_sub = shap_values[:, idx]
    X_sub = X_std[:, idx]
    names_sub = [feature_names[i] for i in idx]

    mean_abs = np.mean(np.abs(shap_sub), axis=0)
    order = np.argsort(-mean_abs)[: min(topk, shap_sub.shape[1])]

    shap_top = shap_sub[:, order]
    X_top = X_sub[:, order]
    names_top = _abbreviate_names([names_sub[i] for i in order])

    plt.figure(figsize=(7.6, 6.2))
    shap.summary_plot(
        shap_top,
        features=X_top,
        feature_names=names_top,
        show=False,
        max_display=len(names_top),
        plot_size=None,
        color_bar=True,
    )
    ax = plt.gca()
    if xlabel:
        ax.set_xlabel(xlabel)

    ax.axvline(0, color="#555", lw=1.2, zorder=0)
    xmax = float(np.quantile(np.abs(shap_top), 0.98))
    xmax = xmax if xmax > 0 else 1.0
    span = 1.1 * xmax
    ax.set_xlim(-span, span)

    if xticks_step is not None and xticks_step > 0:
        hi = np.ceil(span / xticks_step) * xticks_step
        ticks = np.arange(-hi, hi + 1e-12, xticks_step)
        ax.set_xticks(ticks)

    ax.grid(True, axis="x", lw=0.6, color="#ddd", alpha=0.4)
    ensure_dir(Path(out_png).parent)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def plot_topk_bar(shap_values, names, out_png, topk, max_chars=28):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)[:topk]

    names_full = [names[i] for i in order]
    names_plot = [_short_label(name, max_chars=max_chars) for name in names_full][::-1]
    values_plot = mean_abs[order][::-1]

    plt.figure(figsize=(6.0, 4.6))
    plt.barh(names_plot, values_plot)
    plt.xlabel("mean(|SHAP|)")
    ensure_dir(Path(out_png).parent)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return names_full


def plot_dependence(shap_values, X_std, feature_names, feature_idx, out_png):
    x = X_std[:, feature_idx]
    y = shap_values[:, feature_idx]

    plt.figure(figsize=(5.0, 4.2))
    plt.scatter(x, y, s=16, alpha=0.75, edgecolors="none")
    plt.xlabel(f"{feature_names[feature_idx]} (z-score)")
    plt.ylabel("SHAP value")
    ensure_dir(Path(out_png).parent)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def family_of(name):
    name = name.lower()
    if any(token in name for token in ["rotatable", "sp3"]):
        return "flex"
    if any(token in name for token in ["aromatic", "balaban", "sp2", "kappa"]):
        return "rigid"
    if any(token in name for token in ["hbond", "topological_surface_area", "tpsa"]):
        return "polarHB"
    if any(token in name for token in ["smrvsa", "slogpvsa", "molmr"]):
        return "polz"
    if any(token in name for token in ["ring", "bicyclic", "bridging"]):
        return "rings"
    if any(token in name for token in ["halogen", "heteroatom", "no", "nn", "nhal"]):
        return "comp"
    if any(token in name for token in ["labuteasa", "logp"]):
        return "physchem"
    return "other"


def plot_descriptor_quadrants(
    names,
    shap_tc,
    shap_mod,
    pf_idx,
    cons_tc,
    cons_mod,
    out_png,
    out_csv,
    topk_labels=12,
    cons_thr=0.70,
    include_prefixes=("[B]", "[S]"),
    label_radial_push=0.12,
    title="Quadrants (PF)",
):
    if include_prefixes:
        mask = [any(name.startswith(prefix) for prefix in include_prefixes) for name in names]
        if any(mask):
            names = [name for name, keep in zip(names, mask) if keep]
            shap_tc = shap_tc[:, mask]
            shap_mod = shap_mod[:, mask]
            cons_tc = cons_tc[mask] if cons_tc is not None else None
            cons_mod = cons_mod[mask] if cons_mod is not None else None

    phi_tc = np.median(shap_tc[pf_idx], axis=0)
    phi_mod = np.median(shap_mod[pf_idx], axis=0)

    df = pd.DataFrame(
        {
            "Descriptor": names,
            "phi_TC": phi_tc,
            "phi_Modulus": phi_mod,
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def scale(values):
        q75 = float(np.quantile(np.abs(values), 0.75))
        return values / (q75 if q75 > 1e-12 else 1.0)

    df["x"] = np.clip(scale(df["phi_TC"]), -6, 6)
    df["y"] = np.clip(scale(df["phi_Modulus"]), -6, 6)
    df["r"] = np.hypot(df["x"], df["y"])

    quadrant_ideal = "Ideal"
    quadrant_comp = "Compensate"
    quadrant_risk = "Risk"
    quadrant_useless = "Useless"

    def assign_quadrant(row):
        if row["x"] >= 0 and row["y"] <= 0:
            return quadrant_ideal
        if row["x"] < 0 and row["y"] <= 0:
            return quadrant_comp
        if row["x"] >= 0 and row["y"] > 0:
            return quadrant_risk
        return quadrant_useless

    df["Quadrant"] = df.apply(assign_quadrant, axis=1)
    df["family"] = df["Descriptor"].map(family_of)

    if cons_tc is not None:
        df["cons_TC"] = cons_tc
    if cons_mod is not None:
        df["cons_Modulus"] = cons_mod
    df["cons_min"] = np.minimum(df.get("cons_TC", np.nan), df.get("cons_Modulus", np.nan))

    df.sort_values("r", ascending=False).to_csv(out_csv, index=False)

    colors = {
        quadrant_ideal: "#2ca02c",
        quadrant_comp: "#1f77b4",
        quadrant_risk: "#d62728",
        quadrant_useless: "#7f7f7f",
    }
    sizes = {
        quadrant_ideal: 50,
        quadrant_comp: 40,
        quadrant_risk: 34,
        quadrant_useless: 30,
    }
    alphas = {
        quadrant_ideal: 0.95,
        quadrant_comp: 0.90,
        quadrant_risk: 0.80,
        quadrant_useless: 0.80,
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axhline(0, color="#444", lw=1.0, zorder=0)
    ax.axvline(0, color="#444", lw=1.0, zorder=0)

    xlim = (float(np.min(df["x"])) - 3.0, float(np.max(df["x"])) + 0.6)
    ylim = (float(np.min(df["y"])) - 0.6, float(np.max(df["y"])) + 0.6)
    y_bottom = min(ylim[0], 0.0)

    if 0.0 > y_bottom:
        ax.add_patch(
            Rectangle((0.0, y_bottom), xlim[1], -y_bottom, facecolor=colors[quadrant_ideal], alpha=0.07, lw=0, zorder=0)
        )
        ax.add_patch(
            Rectangle((xlim[0], y_bottom), -xlim[0], -y_bottom, facecolor=colors[quadrant_comp], alpha=0.06, lw=0, zorder=0)
        )

    for quadrant in [quadrant_useless, quadrant_risk, quadrant_comp, quadrant_ideal]:
        sub = df[df["Quadrant"] == quadrant]
        if len(sub):
            ax.scatter(
                sub["x"],
                sub["y"],
                s=sizes[quadrant],
                alpha=alphas[quadrant],
                c=colors[quadrant],
                edgecolors="none",
                label=quadrant,
            )

    total_k = max(1, topk_labels)
    k_ideal = max(1, int(np.ceil(total_k * 0.60)))
    k_comp = max(1, int(np.ceil(total_k * 0.30)))
    k_risk = max(0, total_k - k_ideal - k_comp)

    def pick(sub_df, k):
        cand = sub_df.sort_values("r", ascending=False).copy()
        seen = set()
        kept = []
        for _, row in cand.iterrows():
            if row["family"] in seen:
                continue
            kept.append(row)
            seen.add(row["family"])
            if len(kept) >= k:
                break
        return pd.DataFrame(kept)

    labels = pd.concat(
        [
            pick(df[df["Quadrant"] == quadrant_ideal], k_ideal),
            pick(df[df["Quadrant"] == quadrant_comp], k_comp),
            pick(df[df["Quadrant"] == quadrant_risk], k_risk),
        ],
        axis=0,
    ).sort_values("r", ascending=False)

    needed_families = [family for family in ["flex", "rigid", "polarHB", "polz", "rings"] if family not in set(labels["family"])]
    if needed_families:
        picked = set(labels["Descriptor"])
        extras = []
        for _, row in df.sort_values("r", ascending=False).iterrows():
            if row["family"] in needed_families and row["Descriptor"] not in picked:
                extras.append(row)
                picked.add(row["Descriptor"])
                needed_families.remove(row["family"])
            if not needed_families:
                break
        if extras:
            labels = pd.concat([labels, pd.DataFrame(extras)], axis=0).sort_values("r", ascending=False)

    labels = labels.copy().reset_index(drop=True)
    labels["Disp"] = [_short_label(name) for name in labels["Descriptor"]]

    def anchor_and_offset(row):
        quadrant = row["Quadrant"]
        if quadrant == quadrant_ideal:
            return {"ha": "left", "va": "top"}, (0.08, -0.10)
        if quadrant == quadrant_comp:
            return {"ha": "right", "va": "top"}, (-0.08, -0.10)
        if quadrant == quadrant_risk:
            return {"ha": "left", "va": "bottom"}, (0.10, 0.12)
        return {"ha": "right", "va": "bottom"}, (-0.08, 0.10)

    def push_radially(x, y, delta):
        norm = np.hypot(x, y)
        if norm < 1e-8:
            return x + delta, y + delta
        return x + delta * x / norm, y + delta * y / norm

    text_objects = []
    anchors = []
    for _, row in labels.iterrows():
        align, (dx, dy) = anchor_and_offset(row)
        bold = not np.isnan(row["cons_min"]) and row["cons_min"] >= cons_thr
        x_text, y_text = push_radially(row["x"] + dx, row["y"] + dy, label_radial_push)
        text = ax.text(
            x_text,
            y_text,
            f" {row['Disp']}",
            fontsize=9,
            fontweight="bold" if bold else "normal",
            color="black",
            zorder=4,
            **align,
        )
        text_objects.append(text)
        anchors.append((row["x"], row["y"]))

    fig.canvas.draw()
    for _ in range(120):
        moved = False
        renderer = fig.canvas.get_renderer()
        bboxes = [text.get_window_extent(renderer=renderer).expanded(1.02, 1.10) for text in text_objects]
        for i in range(len(text_objects)):
            for j in range(i + 1, len(text_objects)):
                if bboxes[i].overlaps(bboxes[j]):
                    xi, yi = text_objects[i].get_position()
                    xj, yj = text_objects[j].get_position()
                    shift = 0.02
                    text_objects[i].set_position((xi - shift, yi + shift))
                    text_objects[j].set_position((xj + shift, yj - shift))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()

    for (x0, y0), text in zip(anchors, text_objects):
        x1, y1 = text.get_position()
        ax.annotate(
            "",
            xy=(x0, y0),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#666", alpha=0.8),
            zorder=3,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9, frameon=False, labelspacing=0.4)
    ax.set_title(title)
    plt.tight_layout()
    ensure_dir(Path(out_png).parent)
    plt.savefig(out_png)
    plt.close()


def concept_for(feature_name):
    name = feature_name.lower()
    layer = "[B]" if name.startswith("[b]") else "[S]" if name.startswith("[s]") else "[F]" if name.startswith("[f]") else ""
    if any(token in name for token in ["num_rotatable_bonds", "rotbonds", "fractioncsp3", "sp3carbon"]):
        return f"{layer} backbone/sidechain flexibility / saturation"
    if any(token in name for token in ["num_aromatic_rings", "aromaticatomfrac", "sp2carbon", "balaban", "conjbond", "conjbondfrac", "kappa"]):
        return f"{layer} pi-conjugation / rigidity"
    if any(token in name for token in ["num_hbond", "hbond", "tpsa", "amide", "ester", "ether", "hbd", "hba"]):
        return f"{layer} polarity / H-bond capacity"
    if any(token in name for token in ["molmr", "polarizability"]):
        return f"{layer} polarizability"
    if any(token in name for token in ["labuteasa", "vsa", "topological_surface_area", "surface"]):
        return f"{layer} accessible surface / packing"
    if any(token in name for token in ["halogen", "heteroatom", "no", "nn", "nhal"]):
        return f"{layer} heteroatom / composition"
    if any(token in name for token in ["longestpathlen", "bertzct", "kappa", "ring", "bicyclic", "bridging", "maxringsize"]):
        return f"{layer} topology / size-shape / rings"
    return f"{layer} other"


def write_mechanistic_summary(shap_tc, shap_mod, names, out_txt, topk=15, target_transform="None", stability_lines=None):
    def top_lines(shap_values, title):
        mean_abs = pd.Series(np.mean(np.abs(shap_values), axis=0), index=names).sort_values(ascending=False)
        mean_sign = pd.Series(np.mean(shap_values, axis=0), index=names)
        top = list(mean_abs.index[:topk])

        lines = [f"{title} - auto summary (Top-{topk}, OOF SHAP)"]
        for descriptor in top:
            effect = "up" if mean_sign[descriptor] > 0 else "down"
            lines.append(f"- {descriptor} | {concept_for(descriptor)} | effect: {effect}")
        return "\n".join(lines)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(top_lines(shap_tc, "TC"))
        f.write("\n")
        f.write(top_lines(shap_mod, f"Modulus (raw SHAP; trained in transformed domain: {target_transform})"))
        f.write("\n")
        if stability_lines:
            f.write("\n[Stability]\n")
            for line in stability_lines:
                f.write(f"- {line}\n")


def export_topk_csv(shap_values, names, out_csv, topk):
    mean_abs = pd.Series(np.mean(np.abs(shap_values), axis=0), index=names).sort_values(ascending=False)
    mean_sign = pd.Series(np.mean(shap_values, axis=0), index=names).reindex(mean_abs.index)
    pd.DataFrame(
        {
            "Descriptor": mean_abs.index[:topk],
            "mean_|SHAP|": mean_abs.values[:topk],
            "mean_SHAP": mean_sign.values[:topk],
        }
    ).to_csv(out_csv, index=False)


def slugify(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def run_plot_shap_results(results_dir, topk=15, topk_beeswarm=15, cons_thr=0.70):
    results_dir = Path(results_dir)

    with open(results_dir / "names_kept.json", "r", encoding="utf-8") as f:
        names_kept = json.load(f)

    X_kept = np.load(results_dir / "X_kept.npy")
    pf_idx = np.load(results_dir / "pf_idx.npy")
    shap_tc_oof = np.load(results_dir / "shap_oof_TC.npy")
    shap_mod_oof = np.load(results_dir / "shap_oof_MOD.npy")
    cons_tc = np.load(results_dir / "cons_tc.npy")
    cons_mod = np.load(results_dir / "cons_mod.npy")

    stats = {}
    metadata = {}
    if (results_dir / "analysis_stats.json").exists():
        with open(results_dir / "analysis_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
    if (results_dir / "model_metadata.json").exists():
        with open(results_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    target_transform = metadata.get("target_transform", "None")

    plot_beeswarm_filtered(
        shap_tc_oof,
        X_kept,
        names_kept,
        results_dir / "beeswarm_TC_B+S.png",
        topk=topk_beeswarm,
        include_prefixes=("[B]", "[S]"),
        xlabel="SHAP value (impact on model output)",
    )
    plot_beeswarm_filtered(
        shap_mod_oof,
        X_kept,
        names_kept,
        results_dir / "beeswarm_MOD_B+S.png",
        topk=topk_beeswarm,
        include_prefixes=("[B]", "[S]"),
        xlabel="SHAP value (impact on model output)",
        xticks_step=0.05,
    )

    top_tc = plot_topk_bar(shap_tc_oof, names_kept, results_dir / "bar_topK_TC.png", topk)
    top_mod = plot_topk_bar(shap_mod_oof, names_kept, results_dir / "bar_topK_MOD.png", topk)

    for i, feature in enumerate(top_tc[:2], start=1):
        plot_dependence(
            shap_tc_oof,
            X_kept,
            names_kept,
            names_kept.index(feature),
            results_dir / f"dep{i}_TC_{slugify(feature)}.png",
        )
    for i, feature in enumerate(top_mod[:2], start=1):
        plot_dependence(
            shap_mod_oof,
            X_kept,
            names_kept,
            names_kept.index(feature),
            results_dir / f"dep{i}_MOD_{slugify(feature)}.png",
        )

    export_topk_csv(shap_tc_oof, names_kept, results_dir / "shap_topK_TC.csv", topk)
    export_topk_csv(shap_mod_oof, names_kept, results_dir / "shap_topK_MOD.csv", topk)

    plot_descriptor_quadrants(
        names_kept,
        shap_tc_oof,
        shap_mod_oof,
        pf_idx,
        cons_tc,
        cons_mod,
        results_dir / f"desc_quadrants_PF_top{topk}.png",
        results_dir / "desc_quadrants_PF_full.csv",
        topk_labels=topk,
        cons_thr=cons_thr,
        include_prefixes=("[B]", "[S]"),
    )

    consistency = stats.get("consistency_counts", {})
    bootstrap = stats.get("bootstrap", {})
    yscramble = stats.get("yscramble", {})

    stability_lines = [
        (
            f"PF sign-consistency threshold={cons_thr:.2f}; counts (TC/Modulus) with >=thr: "
            f"{consistency.get('tc_ge_thr', 0)}/{consistency.get('tc_total', 0)} , "
            f"{consistency.get('mod_ge_thr', 0)}/{consistency.get('mod_total', 0)}"
        ),
        (
            f"Top-{topk} bootstrap Jaccard (TC/Modulus): "
            f"{bootstrap.get('tc_jaccard_mean', 0.0):.3f} / {bootstrap.get('mod_jaccard_mean', 0.0):.3f}"
        ),
        (
            f"y-scramble Top-{topk} mean overlap (TC/Modulus): "
            f"{yscramble.get('tc_overlap_mean', 0.0):.2f} / {yscramble.get('mod_overlap_mean', 0.0):.2f}; "
            f"Jaccard {yscramble.get('tc_jaccard_mean', 0.0):.3f} / {yscramble.get('mod_jaccard_mean', 0.0):.3f}"
        ),
    ]

    write_mechanistic_summary(
        shap_tc_oof,
        shap_mod_oof,
        names_kept,
        results_dir / "mechanistic_summary.txt",
        topk=topk,
        target_transform=target_transform,
        stability_lines=stability_lines,
    )

    return {"results_dir": str(results_dir), "topk": int(topk), "topk_beeswarm": int(topk_beeswarm)}


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "shap_postanalysis" / "results"
    run_plot_shap_results(results_dir=results_dir, topk=15, topk_beeswarm=15, cons_thr=0.70)


if __name__ == "__main__":
    main()
