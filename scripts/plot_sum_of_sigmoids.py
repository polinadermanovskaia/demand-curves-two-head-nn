"""Generate sum-of-sigmoids illustration for Chapter 2.4."""

import numpy as np
import matplotlib.pyplot as plt

# Relative price range
r = np.linspace(0.7, 1.4, 500)

# Four sigmoid components: g_k(r) = -a_k * sigmoid(b_k * (r - c_k))
# Each contributes a monotone-decreasing step at a different price threshold
components = [
    {"a": 0.8, "b": 15, "c": 0.85, "label": "Component 1 (low-price threshold)"},
    {"a": 1.2, "b": 10, "c": 0.95, "label": "Component 2 (near-baseline)"},
    {"a": 0.6, "b": 12, "c": 1.10, "label": "Component 3 (moderate increase)"},
    {"a": 0.4, "b": 20, "c": 1.25, "label": "Component 4 (high-price threshold)"},
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Left panel: individual sigmoid components
ax = axes[0]
for i, comp in enumerate(components):
    sigmoid = 1 / (1 + np.exp(-comp["b"] * (r - comp["c"])))
    y = -comp["a"] * sigmoid
    ax.plot(r, y, color=colors[i], linewidth=1.8, linestyle="--", label=comp["label"])

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(1.0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Relative price  $r = P/P_0$", fontsize=11)
ax.set_ylabel("Component value", fontsize=11)
ax.set_title("(a) Individual sigmoid components", fontsize=12)
ax.legend(fontsize=8, loc="lower left")

# Right panel: their sum = g_price(r), anchored at g(1) = 0
ax = axes[1]
total = np.zeros_like(r)
for comp in components:
    sigmoid = 1 / (1 + np.exp(-comp["b"] * (r - comp["c"])))
    total += -comp["a"] * sigmoid

# Anchor: shift so g(1) = 0
anchor_idx = np.argmin(np.abs(r - 1.0))
total_anchored = total - total[anchor_idx]

ax.plot(r, total_anchored, color="#2C3E50", linewidth=2.5, label="$g_{\\mathrm{price}}(r)$ (sum)")
ax.scatter([1.0], [0.0], color="#E74C3C", s=80, zorder=5, label="Anchor: $g(1) = 0$")
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(1.0, color="gray", linewidth=0.5, linestyle=":")

# Also show a power-law curve for comparison
power_law = -2.0 * np.log(r)  # log-log with elasticity -2
ax.plot(r, power_law, color="#8E44AD", linewidth=1.5, linestyle=":", label="Power-law $r^{-s}$ (log-log)")

ax.set_xlabel("Relative price  $r = P/P_0$", fontsize=11)
ax.set_ylabel("$g_{\\mathrm{price}}(r)$  (log-demand adjustment)", fontsize=11)
ax.set_title("(b) Sum-of-sigmoids vs. power-law", fontsize=12)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
plt.savefig("results/figures/sum_of_sigmoids_illustration.png", dpi=200, bbox_inches="tight")
print("Saved to results/figures/sum_of_sigmoids_illustration.png")
