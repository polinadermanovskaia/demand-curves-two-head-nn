"""
Generate demand curves by review count quartiles figure.
This script updates the demand_curves_by_review_count.png to use quartiles
(Q1: 0-6, Q2: 7-12, Q3: 13-24, Q4: 25+) instead of the old split (0, 1-5, 5+).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

def load_elasticities():
    """Load pre-computed elasticities."""
    elasticities_df = pd.read_csv(TABLES_DIR / 'product_elasticities.csv')
    return elasticities_df

def create_quartile_groups(elasticities_df):
    """Create quartile groups based on review count."""
    # Use the quartile boundaries from the thesis
    # Q1: 0-6, Q2: 7-12, Q3: 13-24, Q4: 25+
    def quartile_group(count):
        if count <= 6:
            return 'Q1 (0-6)'
        elif count <= 12:
            return 'Q2 (7-12)'
        elif count <= 24:
            return 'Q3 (13-24)'
        else:
            return 'Q4 (25+)'

    elasticities_df['quartile_group'] = elasticities_df['review_count'].apply(quartile_group)
    return elasticities_df

def plot_demand_curves_by_quartile(elasticities_df, n_per_group=4):
    """Plot demand curves for sample products from each quartile."""
    quartile_order = ['Q1 (0-6)', 'Q2 (7-12)', 'Q3 (13-24)', 'Q4 (25+)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Price range for plotting
    r_range = np.linspace(0.785, 1.283, 100)
    log_r = np.log(r_range)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_per_group))

    for idx, (quartile, ax) in enumerate(zip(quartile_order, axes)):
        group_products = elasticities_df[elasticities_df['quartile_group'] == quartile]

        if len(group_products) == 0:
            ax.set_title(f'{quartile}\n(No products)')
            continue

        # Sample products from this group
        sample_size = min(n_per_group, len(group_products))
        sample_products = group_products.sample(sample_size, random_state=42)

        for i, (_, product) in enumerate(sample_products.iterrows()):
            # Create demand curve using elasticity (simplified linear approximation)
            # For a constant elasticity, log Q = ε * log r (anchored at r=1)
            curve = product['elasticity'] * log_r
            # Anchor at r=1 (log_r = 0)
            curve = curve - curve[np.argmin(np.abs(r_range - 1.0))]

            label = f"ε={product['elasticity']:.2f}"
            ax.plot(r_range, curve, color=colors[i], linewidth=2, label=label, alpha=0.8)

        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Relative Price (r)')
        ax.set_ylabel('Log Demand Change')
        ax.set_title(f'{quartile}\nMedian ε = {group_products["elasticity"].median():.2f}')
        ax.legend(fontsize=8, loc='lower left')
        ax.set_xlim(0.75, 1.35)

    plt.suptitle('Demand Curves by Review Count Quartile', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig

def main():
    print("Loading elasticities data...")
    elasticities_df = load_elasticities()

    print("Creating quartile groups...")
    elasticities_df = create_quartile_groups(elasticities_df)

    print("\nQuartile distribution:")
    print(elasticities_df['quartile_group'].value_counts().sort_index())

    print("\nGenerating demand curves plot...")
    fig = plot_demand_curves_by_quartile(elasticities_df, n_per_group=4)

    output_path = FIGURES_DIR / 'demand_curves_by_review_count.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    plt.close()
    print("Done!")

if __name__ == '__main__':
    main()
