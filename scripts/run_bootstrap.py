"""
Re-run clustered bootstrap CI using the canonical NN model (improved_nn_model.pt).
Saves results to tables/bootstrap_ci_results.csv.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path
import time
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR = Path('data/processed')
TABLES_DIR = Path('results/tables')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Feature definitions ──────────────────────────────────────────────────────
CONTEXT_FEATURES = [
    'year', 'month', 'weekofyear', 'week_sin', 'week_cos',
    'demand_lag_1', 'demand_lag_2', 'demand_roll_4',
    'price_lag_1', 'price_roll_4',
    'weeks_since_last_sale',
    'price_std', 'price_range'
]
PRODUCT_FEATURES = [
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
    'product_photos_qty', 'product_name_length', 'product_description_length'
]
REVIEW_FEATURES = ['sku_review_count', 'sku_review_mean', 'sku_share_low']
PRICE_FEATURE = 'r_clipped'
TARGET = 'y'

# ── Load data ────────────────────────────────────────────────────────────────
panel = pd.read_csv(DATA_DIR / 'panel.csv')

for col in ['demand_lag_1', 'demand_lag_2', 'demand_roll_4', 'price_lag_1', 'price_roll_4', 'weeks_since_last_sale']:
    panel[col] = panel[col].fillna(0)
for col in REVIEW_FEATURES:
    panel[col] = panel[col].fillna(0)
for col in PRODUCT_FEATURES:
    panel[col] = panel[col].fillna(panel[col].median())

le_category = LabelEncoder()
panel['category_code'] = le_category.fit_transform(panel['product_category_name_english'].fillna('unknown'))
n_categories = len(le_category.classes_)

le_product = LabelEncoder()
panel['product_code'] = le_product.fit_transform(panel['product_id'])
n_products = len(le_product.classes_)

train_df = panel[panel['split'] == 'train'].copy()
val_df   = panel[panel['split'] == 'val'].copy()
test_df  = panel[panel['split'] == 'test'].copy()

scaler_context = StandardScaler()
scaler_product = StandardScaler()
scaler_review  = StandardScaler()

train_context = np.nan_to_num(scaler_context.fit_transform(train_df[CONTEXT_FEATURES]), nan=0.0)
train_product = np.nan_to_num(scaler_product.fit_transform(train_df[PRODUCT_FEATURES]), nan=0.0)
train_review  = np.nan_to_num(scaler_review.fit_transform(train_df[REVIEW_FEATURES]),  nan=0.0)

val_context = np.nan_to_num(scaler_context.transform(val_df[CONTEXT_FEATURES]), nan=0.0)
val_product = np.nan_to_num(scaler_product.transform(val_df[PRODUCT_FEATURES]), nan=0.0)
val_review  = np.nan_to_num(scaler_review.transform(val_df[REVIEW_FEATURES]),  nan=0.0)

test_context = np.nan_to_num(scaler_context.transform(test_df[CONTEXT_FEATURES]), nan=0.0)
test_product = np.nan_to_num(scaler_product.transform(test_df[PRODUCT_FEATURES]), nan=0.0)
test_review  = np.nan_to_num(scaler_review.transform(test_df[REVIEW_FEATURES]),  nan=0.0)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── Model definitions ────────────────────────────────────────────────────────
class DemandDataset(Dataset):
    def __init__(self, df, context_arr, product_arr, review_arr):
        self.context    = torch.FloatTensor(context_arr)
        self.product    = torch.FloatTensor(product_arr)
        self.review     = torch.FloatTensor(review_arr)
        self.r          = torch.FloatTensor(df[PRICE_FEATURE].values)
        self.log_r      = torch.log(self.r)
        self.category   = torch.LongTensor(df['category_code'].values)
        self.product_id = torch.LongTensor(df['product_code'].values)
        self.y          = torch.FloatTensor(df[TARGET].values)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {
            'context': self.context[idx], 'product': self.product[idx],
            'review': self.review[idx], 'r': self.r[idx], 'log_r': self.log_r[idx],
            'category': self.category[idx], 'product_id': self.product_id[idx],
            'y': self.y[idx]
        }

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.net(x)

class MonotonicPriceEncoder(nn.Module):
    def __init__(self, z_dim, num_basis=20, hidden_dim=128):
        super().__init__()
        self.num_basis = num_basis
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3 * num_basis)
        )
    def forward(self, log_r, z, return_at_ref=False):
        B = log_r.shape[0]
        params = self.z_encoder(z).view(B, self.num_basis, 3)
        a = F.softplus(params[:, :, 0]) + 0.1
        b = params[:, :, 1]
        w = -F.softplus(params[:, :, 2])
        g = (w * torch.sigmoid(a * log_r.unsqueeze(1) + b)).sum(dim=1, keepdim=True)
        if return_at_ref:
            g_ref = (w * torch.sigmoid(b)).sum(dim=1, keepdim=True)
            return g, g_ref
        return g

class ImprovedTwoHeadModel(nn.Module):
    def __init__(self, context_dim, product_dim, review_dim, n_categories, n_products,
                 category_embedding_dim=16, product_embedding_dim=16,
                 context_hidden=256, price_hidden=128, num_basis=20, dropout=0.15):
        super().__init__()
        self.category_embedding = nn.Embedding(n_categories, category_embedding_dim)
        self.product_embedding  = nn.Embedding(n_products, product_embedding_dim)
        context_input_dim = context_dim + category_embedding_dim + product_embedding_dim
        self.context_encoder = ContextEncoder(context_input_dim, context_hidden, dropout)
        z_dim = category_embedding_dim + product_embedding_dim + product_dim + review_dim
        self.price_encoder = MonotonicPriceEncoder(z_dim, num_basis, price_hidden)
    def forward(self, batch):
        cat_emb  = self.category_embedding(batch['category'])
        prod_emb = self.product_embedding(batch['product_id'])
        ctx_in   = torch.cat([batch['context'], cat_emb, prod_emb], dim=1)
        log_base = self.context_encoder(ctx_in)
        z        = torch.cat([cat_emb, prod_emb, batch['product'], batch['review']], dim=1)
        g_r, g_ref = self.price_encoder(batch['log_r'], z, return_at_ref=True)
        return (log_base + g_r - g_ref).squeeze(1)

# ── Evaluation helpers ───────────────────────────────────────────────────────
def compute_corr_mse(y_true, y_pred, groups):
    unique_groups = np.unique(groups)
    group_mses, group_weights = [], []
    for g in unique_groups:
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        a_opt = yt.mean() - yp.mean()
        group_mses.append(np.mean((yt - (yp + a_opt)) ** 2))
        group_weights.append(len(yt))
    return np.average(group_mses, weights=group_weights)

def evaluate_model(model, loader, df):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_preds.append(model(batch).cpu().numpy())
            all_targets.append(batch['y'].cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    corr_mse = compute_corr_mse(y_true, y_pred, df['product_id'].values)
    return {'corr_mse': corr_mse, 'r2': r2_score(y_true, y_pred)}, y_pred

# ── Data loaders ─────────────────────────────────────────────────────────────
BATCH_SIZE = 256
test_dataset  = DemandDataset(test_df,  test_context,  test_product,  test_review)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Step 1: Per-product log-log predictions ──────────────────────────────────
usable_products = panel[panel['usable_for_elasticity'] == 1]['product_id'].unique()
global_mean_y = train_df['y'].mean()
loglog_test_preds = np.full(len(test_df), global_mean_y)

for pid in usable_products:
    train_mask = train_df['product_id'] == pid
    if train_mask.sum() < 3:
        continue
    X_tr = np.log(train_df.loc[train_mask, 'r_clipped'].values).reshape(-1, 1)
    y_tr = train_df.loc[train_mask, 'y'].values
    lr = LinearRegression().fit(X_tr, y_tr)
    test_mask = test_df['product_id'] == pid
    if test_mask.sum() == 0:
        continue
    X_te = np.log(test_df.loc[test_mask, 'r_clipped'].values).reshape(-1, 1)
    pos_idx = np.where(test_mask.values)[0]
    loglog_test_preds[pos_idx] = lr.predict(X_te)

loglog_corr_mse = compute_corr_mse(test_df['y'].values, loglog_test_preds, test_df['product_id'].values)
print(f"Per-product log-log corr_mse: {loglog_corr_mse:.4f}  (expected ~0.1086)")

# ── Step 2: Load canonical NN model ─────────────────────────────────────────
config = {
    'context_dim': len(CONTEXT_FEATURES), 'product_dim': len(PRODUCT_FEATURES),
    'review_dim': len(REVIEW_FEATURES), 'n_categories': n_categories,
    'n_products': n_products, 'category_embedding_dim': 16, 'product_embedding_dim': 16,
    'context_hidden': 256, 'price_hidden': 128, 'num_basis': 20, 'dropout': 0.15
}

checkpoint = torch.load(Path('results') / 'improved_nn_model.pt', map_location=device, weights_only=False)
model = ImprovedTwoHeadModel(**config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

nn_metrics, nn_test_preds = evaluate_model(model, test_loader, test_df)
print(f"NN corr_mse: {nn_metrics['corr_mse']:.4f}  (expected ~0.1080)")
print(f"Point diff (NN - log-log): {nn_metrics['corr_mse'] - loglog_corr_mse:.4f}")

# ── Step 3: XGBoost predictions ──────────────────────────────────────────────
for df_part in [train_df, val_df, test_df]:
    df_part['log_r'] = np.log(df_part['r_clipped'].values)

ALL_FEATURES_XGB = ['log_r'] + CONTEXT_FEATURES + PRODUCT_FEATURES + REVIEW_FEATURES
dtrain = xgb.DMatrix(train_df[ALL_FEATURES_XGB].values, label=train_df['y'].values, feature_names=ALL_FEATURES_XGB)
dval   = xgb.DMatrix(val_df[ALL_FEATURES_XGB].values,   label=val_df['y'].values,   feature_names=ALL_FEATURES_XGB)
dtest  = xgb.DMatrix(test_df[ALL_FEATURES_XGB].values,                               feature_names=ALL_FEATURES_XGB)

xgb_params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.1,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
               'seed': 42, 'verbosity': 0}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                       evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
xgb_test_preds = xgb_model.predict(dtest)
xgb_corr_mse = compute_corr_mse(test_df['y'].values, xgb_test_preds, test_df['product_id'].values)
print(f"XGBoost corr_mse: {xgb_corr_mse:.4f}  (expected ~0.0964)")

# ── Step 4: Clustered bootstrap ──────────────────────────────────────────────
def compute_corr_mse_for_products(y_true, y_pred, product_ids, selected_products):
    total_mse, total_weight = 0.0, 0
    for pid in selected_products:
        mask = product_ids == pid
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            continue
        a_opt = yt.mean() - yp.mean()
        total_mse += np.mean((yt - (yp + a_opt)) ** 2) * len(yt)
        total_weight += len(yt)
    return total_mse / total_weight if total_weight > 0 else 0.0

y_true_test       = test_df['y'].values
product_ids_test  = test_df['product_id'].values
unique_products   = np.unique(product_ids_test)
n_products_test   = len(unique_products)
N_BOOTSTRAP       = 1000

print(f"\nBootstrapping {N_BOOTSTRAP} iterations over {n_products_test} products...")
t0 = time.time()

rng = np.random.RandomState(42)
bootstrap_diffs_ll  = np.zeros(N_BOOTSTRAP)
bootstrap_diffs_xgb = np.zeros(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    sampled = rng.choice(unique_products, size=n_products_test, replace=True)
    cmse_nn  = compute_corr_mse_for_products(y_true_test, nn_test_preds,    product_ids_test, sampled)
    cmse_ll  = compute_corr_mse_for_products(y_true_test, loglog_test_preds, product_ids_test, sampled)
    cmse_xgb = compute_corr_mse_for_products(y_true_test, xgb_test_preds,   product_ids_test, sampled)
    bootstrap_diffs_ll[b]  = cmse_nn - cmse_ll
    bootstrap_diffs_xgb[b] = cmse_nn - cmse_xgb

print(f"Done in {time.time()-t0:.1f}s")

ci_lower_ll,  ci_upper_ll  = np.percentile(bootstrap_diffs_ll,  [2.5, 97.5])
ci_lower_xgb, ci_upper_xgb = np.percentile(bootstrap_diffs_xgb, [2.5, 97.5])

# ── Step 5: Save results ─────────────────────────────────────────────────────
summary = pd.DataFrame({
    'Comparison':        ['NN vs Per-product log-log', 'NN vs XGBoost'],
    'corr_mse_NN':       [nn_metrics['corr_mse'], nn_metrics['corr_mse']],
    'corr_mse_Baseline': [loglog_corr_mse, xgb_corr_mse],
    'Diff (NN-Baseline)': [
        nn_metrics['corr_mse'] - loglog_corr_mse,
        nn_metrics['corr_mse'] - xgb_corr_mse
    ],
    'CI_lower':    [ci_lower_ll,  ci_lower_xgb],
    'CI_upper':    [ci_upper_ll,  ci_upper_xgb],
    'Significant': [
        'No' if ci_lower_ll  <= 0 <= ci_upper_ll  else 'Yes',
        'No' if ci_lower_xgb <= 0 <= ci_upper_xgb else 'Yes'
    ]
})

print("\nB1 SUMMARY: Bootstrap 95% CI for corr_mse differences")
print("="*70)
print(summary.to_string(index=False))
summary.to_csv(TABLES_DIR / 'bootstrap_ci_results.csv', index=False)
print(f"\nSaved to {TABLES_DIR / 'bootstrap_ci_results.csv'}")
