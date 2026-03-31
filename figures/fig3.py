import matplotlib.pyplot as plt
improvement = [
    0.8705 - 0.8050,   # 64 → 128
    0.9005 - 0.8705    # 128 → 256
]

labels = ["64 → 128", "128 → 256"]

plt.figure()
plt.bar(labels, improvement)

plt.ylabel("Accuracy Gain")
plt.title("Diminishing Returns in IMDb Accuracy with Increasing Context Length")
plt.grid(axis='y')

plt.savefig("fig_diminishing_returns.png", dpi=300)
plt.show()
