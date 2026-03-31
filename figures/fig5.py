import matplotlib.pyplot as plt
normalized = [
    18.1/64,
    32.4/128,
    53.2/256
]

plt.figure()
plt.plot(lengths, normalized, marker='o')

plt.xlabel("Context Length")
plt.ylabel("Normalized Attention Distance")
plt.title("Relative Attention Span Decreases with Context Length")
plt.grid()

plt.savefig("fig_attention_normalized.png", dpi=300)
plt.show()
