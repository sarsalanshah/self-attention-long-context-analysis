import matplotlib.pyplot as plt
lengths = [64, 128, 256]
distance = [18.1, 32.4, 53.2]

plt.figure()
plt.plot(lengths, distance, marker='o')

plt.xlabel("Context Length")
plt.ylabel("Average Attention Distance")
plt.title("Attention Distance vs Context Length")
plt.grid()

plt.savefig("fig_attention_absolute.png", dpi=300)
plt.show()
