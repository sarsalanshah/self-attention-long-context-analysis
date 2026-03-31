import matplotlib.pyplot as plt

lengths = [64, 128, 256]
accuracy = [0.8050, 0.8705, 0.9005]
f1 = [0.8142, 0.8730, 0.9031]

plt.figure()
plt.plot(lengths, accuracy, marker='o', label='Accuracy')
plt.plot(lengths, f1, marker='s', label='F1 Score')

plt.xlabel("Context Length")
plt.ylabel("Score")
plt.title("IMDb Classification Performance vs Context Length")
plt.legend()
plt.grid()

plt.savefig("fig_imdb_performance.png", dpi=300)
plt.show()
