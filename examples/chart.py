"""Minimal chart used by the README quick start."""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
ax.bar([0, 1, 2], [2, 4, 3], color=["#7094E1", "#9966C3", "#FF7F0E"])  #1
ax.set_xticks([0, 1, 2], ["A", "B", "C"])
ax.set_ylabel("Value")
fig.tight_layout()
