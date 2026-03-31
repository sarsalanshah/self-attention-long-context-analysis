import matplotlib.pyplot as plt

# Define lengths (this was missing)
lengths = [64, 128, 256]

# Your attention values
distance = [18.1, 32.4, 53.2]

# Normalize
normalized = [
    distance[0]/lengths[0],
    distance[1]/lengths[1],
    distance[2]/lengths[2]
]

plt.figure()
plt.plot(lengths, normalized, marker='o')

plt.xlabel("Context Length")
plt.ylabel("Normalized Attention Distance")
plt.title("Relative Attention Span Decreases with Context Length")
plt.grid()

plt.savefig("fig_attention_normalized.png", dpi=300)
plt.show()
