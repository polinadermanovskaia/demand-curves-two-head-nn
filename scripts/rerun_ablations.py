"""
Re-run ablation studies (reviews + freight) with the current model config.
Training settings match robustness_experiments.ipynb: patience=40, epochs=200.

Reviews are evaluated as a paired five-seed ablation. Freight remains a
single-seed (seed 42) ablation. Saves:
  - tables/ablation_reviews_by_seed.csv
  - tables/ablation_reviews.csv
  - tables/ablation_freight.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from pathlib import Path
import time

DATA_DIR  = Path(__file__).parent.parent / "data"  / "processed"
TABLES_DIR = Path(__file__).parent.parent / "results" / "tables"

SEEDS = [42, 123, 456, 789, 2024]
FREIGHT_SEED = 42

# ── Feature lists ─────────────────────────────────────────────────────────────
CONTEXT_FEATURES = [
    "year", "month", "weekofyear", "week_sin", "week_cos",
    "demand_lag_1", "demand_lag_2", "demand_roll_4",
    "price_lag_1", "price_roll_4",
    "weeks_since_last_sale",
    "price_std", "price_range",
]
PRODUCT_FEATURES = [
    "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "product_photos_qty",
    "product_name_length", "product_description_length",
]
REVIEW_FEATURES  = ["sku_review_count", "sku_review_mean", "sku_share_low"]
PRICE_FEATURE    = "r_clipped"
TARGET           = "y"


# ── Dataset ───────────────────────────────────────────────────────────────────
class DemandDataset(Dataset):
    def __init__(self, df, context, product, review):
        self.df      = df.reset_index(drop=True)
        self.context = context.astype(np.float32)
        self.product = product.astype(np.float32)
        self.review  = review.astype(np.float32)   # may be (N, 0) for no-review models
        self.price   = df[PRICE_FEATURE].values.astype(np.float32)
        self.target  = df[TARGET].values.astype(np.float32)
        self.prod_id = df["product_id_code"].values.astype(np.int64)
        self.cat_id  = df["category_code"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "context":     torch.tensor(self.context[idx]),
            "product":     torch.tensor(self.product[idx]),
            "price":       torch.tensor([self.price[idx]]),
            "target":      torch.tensor(self.target[idx]),
            "product_id":  torch.tensor(self.prod_id[idx]),
            "category_id": torch.tensor(self.cat_id[idx]),
        }
        if self.review.shape[1] > 0:
            item["review"] = torch.tensor(self.review[idx])
        else:
            item["review"] = torch.zeros(0)
        return item


# ── Architecture ──────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MonotonicPriceEncoder(nn.Module):
    def __init__(self, z_dim, num_basis=20, hidden_dim=128):
        super().__init__()
        self.num_basis = num_basis
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3 * num_basis),
        )
    def forward(self, log_r, z, return_at_ref=False):
        # Monotone-decreasing price head (matches run_bootstrap.py / robustness_experiments):
        # weights w < 0 via -softplus and slopes a > 0 via softplus, applied to log r.
        B        = log_r.shape[0]
        params   = self.z_encoder(z).view(B, self.num_basis, 3)
        a        = F.softplus(params[:, :, 0]) + 0.1   # slopes > 0
        b        = params[:, :, 1]
        w        = -F.softplus(params[:, :, 2])        # weights < 0 so dg/d(log r) <= 0
        g        = (w * torch.sigmoid(a * log_r.unsqueeze(1) + b)).sum(dim=1)
        if return_at_ref:
            g_ref = (w * torch.sigmoid(b)).sum(dim=1)   # value at r = 1 (log_r = 0)
            return g, g_ref
        return g


class TwoHeadModel(nn.Module):
    def __init__(self, context_dim, product_dim, review_dim,
                 n_categories, n_products,
                 category_embedding_dim=16, product_embedding_dim=16,
                 context_hidden=256, price_hidden=128, num_basis=20, dropout=0.15):
        super().__init__()
        self.category_embedding = nn.Embedding(n_categories, category_embedding_dim)
        self.product_embedding  = nn.Embedding(n_products,   product_embedding_dim)
        ctx_input_dim = context_dim + category_embedding_dim + product_embedding_dim
        self.context_encoder = ContextEncoder(ctx_input_dim, context_hidden, dropout)
        z_dim = category_embedding_dim + product_embedding_dim + product_dim + review_dim
        self.price_encoder = MonotonicPriceEncoder(z_dim, num_basis, price_hidden)

    def forward(self, batch):
        cat_emb  = self.category_embedding(batch["category_id"])
        prod_emb = self.product_embedding(batch["product_id"])
        ctx_in   = torch.cat([batch["context"], cat_emb, prod_emb], dim=-1)
        baseline = self.context_encoder(ctx_in)
        z_parts  = [cat_emb, prod_emb, batch["product"]]
        if batch["review"].shape[-1] > 0:
            z_parts.append(batch["review"])
        z        = torch.cat(z_parts, dim=-1)
        log_r    = torch.log(batch["price"].squeeze(-1))   # batch["price"] is r_clipped
        g_r, g_ref = self.price_encoder(log_r, z, return_at_ref=True)
        return baseline + g_r - g_ref                       # anchored: g = 0 at r = 1


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_corr_mse(y_true, y_pred, groups):
    df = pd.DataFrame({"t": y_true, "p": y_pred, "g": groups})
    total, wt = 0.0, 0.0
    for _, g in df.groupby("g"):
        # Include singleton groups (match run_bootstrap.py / multi-seed convention):
        # a single observation has zero residual after the optimal shift.
        shift = (g["t"] - g["p"]).mean()
        total += ((g["t"] - g["p"] - shift) ** 2).mean() * len(g)
        wt    += len(g)
    return total / wt if wt else float("nan")


def evaluate(model, loader, df, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(batch).cpu().numpy())
            targets.extend(batch["target"].cpu().numpy())
    y_true, y_pred = np.array(targets), np.array(preds)
    return {
        "corr_mse": compute_corr_mse(y_true, y_pred, df["product_id"].values),
        "r2": r2_score(y_true, y_pred),
    }


def train(model, tr_ld, vl_ld, tr_df, vl_df, device,
          epochs=200, lr=5e-4, wd=1e-5, patience=40, clip=1.0):
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()
    best_val, best_state, counter = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        for batch in tr_ld:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = crit(model(batch), batch["target"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        sch.step()
        vm = evaluate(model, vl_ld, vl_df, device)
        if vm["corr_mse"] < best_val:
            best_val   = vm["corr_mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"  early stop @ epoch {ep+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, ep + 1


# ── Data loading ──────────────────────────────────────────────────────────────
def load_panel():
    panel = pd.read_csv(DATA_DIR / "panel.csv")
    lag_cols = ["demand_lag_1", "demand_lag_2", "demand_roll_4",
                "price_lag_1", "price_roll_4", "weeks_since_last_sale"]
    for c in lag_cols + REVIEW_FEATURES:
        panel[c] = panel[c].fillna(0)
    for c in PRODUCT_FEATURES:
        panel[c] = panel[c].fillna(panel[c].median())
    panel["freight_mean"] = panel["freight_mean"].fillna(panel["freight_mean"].median())

    le_cat  = LabelEncoder()
    le_prod = LabelEncoder()
    panel["category_code"]   = le_cat.fit_transform(
        panel["product_category_name_english"].fillna("unknown"))
    panel["product_id_code"] = le_prod.fit_transform(panel["product_id"])

    tr = panel[panel["split"] == "train"].copy()
    vl = panel[panel["split"] == "val"].copy()
    te = panel[panel["split"] == "test"].copy()
    return tr, vl, te, panel["category_code"].nunique(), panel["product_id_code"].nunique()


def run_variant(label, context_feats, review_feats, seed,
                tr_df, vl_df, te_df, n_cat, n_prod, device):
    """Train one model variant, return (n_params, test_metrics, n_epochs, time_s)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    sc_ctx  = StandardScaler()
    sc_prod = StandardScaler()

    tr_ctx  = np.nan_to_num(sc_ctx.fit_transform(tr_df[context_feats]),  nan=0.0).astype(np.float32)
    tr_prod = np.nan_to_num(sc_prod.fit_transform(tr_df[PRODUCT_FEATURES]), nan=0.0).astype(np.float32)
    vl_ctx  = np.nan_to_num(sc_ctx.transform(vl_df[context_feats]),  nan=0.0).astype(np.float32)
    vl_prod = np.nan_to_num(sc_prod.transform(vl_df[PRODUCT_FEATURES]), nan=0.0).astype(np.float32)
    te_ctx  = np.nan_to_num(sc_ctx.transform(te_df[context_feats]),  nan=0.0).astype(np.float32)
    te_prod = np.nan_to_num(sc_prod.transform(te_df[PRODUCT_FEATURES]), nan=0.0).astype(np.float32)

    if review_feats:
        sc_rev  = StandardScaler()
        tr_rev  = np.nan_to_num(sc_rev.fit_transform(tr_df[review_feats]), nan=0.0).astype(np.float32)
        vl_rev  = np.nan_to_num(sc_rev.transform(vl_df[review_feats]), nan=0.0).astype(np.float32)
        te_rev  = np.nan_to_num(sc_rev.transform(te_df[review_feats]), nan=0.0).astype(np.float32)
        rev_dim = len(review_feats)
    else:
        # review_dim = 0 → smaller model, no review input
        tr_rev = vl_rev = te_rev = np.zeros((0,), dtype=np.float32)
        # Actually need shape (N, 0)
        tr_rev = np.empty((len(tr_df), 0), dtype=np.float32)
        vl_rev = np.empty((len(vl_df), 0), dtype=np.float32)
        te_rev = np.empty((len(te_df), 0), dtype=np.float32)
        rev_dim = 0

    gen = torch.Generator().manual_seed(seed)
    tr_ds  = DemandDataset(tr_df, tr_ctx, tr_prod, tr_rev)
    vl_ds  = DemandDataset(vl_df, vl_ctx, vl_prod, vl_rev)
    te_ds  = DemandDataset(te_df, te_ctx, te_prod, te_rev)
    tr_ld  = DataLoader(tr_ds, 256, shuffle=True,  generator=gen)
    vl_ld  = DataLoader(vl_ds, 256, shuffle=False)
    te_ld  = DataLoader(te_ds, 256, shuffle=False)

    model   = TwoHeadModel(len(context_feats), len(PRODUCT_FEATURES), rev_dim, n_cat, n_prod)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}, seed={seed}]  context={len(context_feats)}  "
          f"review_dim={rev_dim}  params={n_params:,}")

    t0 = time.time()
    model, n_ep = train(model, tr_ld, vl_ld, tr_df, vl_df, device)
    elapsed = time.time() - t0

    metrics = evaluate(model, te_ld, te_df, device)
    print(f"  corr_mse={metrics['corr_mse']:.5f}  r2={metrics['r2']:.4f}  "
          f"epochs={n_ep}  time={elapsed:.0f}s")
    return n_params, metrics, n_ep, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_df, vl_df, te_df, n_cat, n_prod = load_panel()
    print(f"Data: train={len(tr_df)}, val={len(vl_df)}, test={len(te_df)}")
    print(f"      categories={n_cat}, products={n_prod}")

    # ── Ablation 1: Reviews ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION 1: Reviews (with vs without)")
    print("="*60)

    review_rows = []
    for seed in SEEDS:
        n_w, m_w, ep_w, time_w = run_variant(
            "with reviews", CONTEXT_FEATURES, REVIEW_FEATURES, seed,
            tr_df, vl_df, te_df, n_cat, n_prod, device
        )
        n_wo, m_wo, ep_wo, time_wo = run_variant(
            "without reviews", CONTEXT_FEATURES, None, seed,
            tr_df, vl_df, te_df, n_cat, n_prod, device
        )
        delta = m_wo["corr_mse"] - m_w["corr_mse"]
        review_rows.append({
            "seed": seed,
            "with_reviews_params": n_w,
            "without_reviews_params": n_wo,
            "with_reviews_corr_mse": m_w["corr_mse"],
            "without_reviews_corr_mse": m_wo["corr_mse"],
            "paired_delta_without_minus_with": delta,
            "paired_delta_pct_without_minus_with": delta / m_w["corr_mse"] * 100,
            "with_reviews_r2": m_w["r2"],
            "without_reviews_r2": m_wo["r2"],
            "with_reviews_epochs": ep_w,
            "without_reviews_epochs": ep_wo,
            "with_reviews_time_s": time_w,
            "without_reviews_time_s": time_wo,
        })

    reviews_by_seed = pd.DataFrame(review_rows)
    reviews_by_seed.to_csv(TABLES_DIR / "ablation_reviews_by_seed.csv", index=False)

    mean_delta = reviews_by_seed["paired_delta_without_minus_with"].mean()
    sd_delta = reviews_by_seed["paired_delta_without_minus_with"].std(ddof=1)
    without_better = int((reviews_by_seed["paired_delta_without_minus_with"] < 0).sum())
    summary_common = {
        "Mean paired delta (without - with)": mean_delta,
        "SD paired delta": sd_delta,
        "Seeds without reviews better": without_better,
        "Total seeds": len(SEEDS),
    }
    reviews_summary = pd.DataFrame([
        {
            "Model": "With reviews",
            "Parameters": int(reviews_by_seed["with_reviews_params"].iloc[0]),
            "Mean test corr_mse": reviews_by_seed["with_reviews_corr_mse"].mean(),
            "SD test corr_mse": reviews_by_seed["with_reviews_corr_mse"].std(ddof=1),
            "Mean test R²": reviews_by_seed["with_reviews_r2"].mean(),
            **summary_common,
        },
        {
            "Model": "Without reviews",
            "Parameters": int(reviews_by_seed["without_reviews_params"].iloc[0]),
            "Mean test corr_mse": reviews_by_seed["without_reviews_corr_mse"].mean(),
            "SD test corr_mse": reviews_by_seed["without_reviews_corr_mse"].std(ddof=1),
            "Mean test R²": reviews_by_seed["without_reviews_r2"].mean(),
            **summary_common,
        },
    ])
    reviews_summary.to_csv(TABLES_DIR / "ablation_reviews.csv", index=False)
    print("\nReview ablation summary:")
    print(reviews_summary.to_string(index=False))
    print(f"\n→ ablation_reviews_by_seed.csv and ablation_reviews.csv saved  "
          f"(mean paired Δ = {mean_delta:+.6f}; without reviews better in "
          f"{without_better}/{len(SEEDS)} seeds)")

    # ── Ablation 2: Freight ──────────────────────────────────────────────────
    CONTEXT_FREIGHT = CONTEXT_FEATURES + ["freight_mean"]

    print("\n" + "="*60)
    print("ABLATION 2: Freight (without vs with)")
    print("="*60)

    # "without freight" is the standard model = seed-42 "with reviews" run above.
    seed_42 = reviews_by_seed.loc[reviews_by_seed["seed"] == FREIGHT_SEED].iloc[0]
    n_nf = int(seed_42["with_reviews_params"])
    m_nf = {
        "corr_mse": seed_42["with_reviews_corr_mse"],
        "r2": seed_42["with_reviews_r2"],
    }
    print(f"\n[without freight]  reusing 'with reviews' seed-{FREIGHT_SEED} run")
    print(f"  corr_mse={m_nf['corr_mse']:.5f}  r2={m_nf['r2']:.4f}")

    n_f, m_f, _, _ = run_variant(
        "with freight", CONTEXT_FREIGHT, REVIEW_FEATURES, FREIGHT_SEED,
        tr_df, vl_df, te_df, n_cat, n_prod, device
    )

    delta_fr  = m_f["corr_mse"] - m_nf["corr_mse"]
    delta_pct2 = delta_fr / m_nf["corr_mse"] * 100

    pd.DataFrame([
        {"experiment": "Without freight (baseline)", "n_features": len(CONTEXT_FEATURES),
         "n_params": n_nf, "test_corr_mse": m_nf["corr_mse"], "test_r2": m_nf["r2"]},
        {"experiment": "With freight",               "n_features": len(CONTEXT_FREIGHT),
         "n_params": n_f,  "test_corr_mse": m_f["corr_mse"],  "test_r2": m_f["r2"]},
    ]).to_csv(TABLES_DIR / "ablation_freight.csv", index=False)
    print(f"\n→ ablation_freight.csv saved  (Δ = {delta_pct2:+.1f}%)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"With reviews    (5-seed mean): corr_mse="
          f"{reviews_by_seed['with_reviews_corr_mse'].mean():.5f}  R²="
          f"{reviews_by_seed['with_reviews_r2'].mean():+.4f}")
    print(f"Without reviews (5-seed mean): corr_mse="
          f"{reviews_by_seed['without_reviews_corr_mse'].mean():.5f}  R²="
          f"{reviews_by_seed['without_reviews_r2'].mean():+.4f}  "
          f"mean paired Δ={mean_delta:+.5f}")
    print(f"With freight    (with reviews): corr_mse={m_f['corr_mse']:.5f}  R²={m_f['r2']:+.4f}  "
          f"Δ={delta_pct2:+.1f}%")


if __name__ == "__main__":
    main()
