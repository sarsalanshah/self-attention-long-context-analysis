import matplotlib.pyplot as plt

lengths = [64, 128, 256]
perplexity = [144.85, 123.37, 118.01]

plt.figure()
plt.plot(lengths, perplexity, marker='o')

plt.xlabel("Context Length")
plt.ylabel("Perplexity")
plt.title("GPT-2 Perplexity vs Context Length")
plt.grid()

plt.savefig("fig_gpt2_perplexity.png", dpi=300)
plt.show()
