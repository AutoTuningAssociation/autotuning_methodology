import matplotlib.pyplot as plt
import numpy as np

# plot individual search space
plt.rcParams["figure.figsize"] = (5, 4)
fevals = np.arange(1, 100)
plt.plot(fevals, fevals**0.5, label="algorithm X")
plt.plot(fevals, fevals**0.4 + np.sin(fevals**0.1), label="algorithm Y")
plt.xlabel("time in seconds")
plt.ylabel("performance\n(higher is better)")
plt.legend()
plt.tight_layout()
plt.savefig("flowchart_plot_individual", dpi=300, bbox_inches="tight")
plt.cla()


def min_max(array):
    return (array - np.min(array)) / (np.max(array) / np.min(array))


# plot aggregated
fevals = np.arange(1, 100)
plt.axhline(y=1, linestyle="-.", color="black", label="Absolute optimum")
plt.axhline(y=0, linestyle=":", color="black", label="Random search")
plt.plot(fevals, min_max(fevals**0.5), label="algorithm X")
plt.plot(fevals, min_max(1.8 + np.sin(fevals**0.5)), label="algorithm Y")
plt.xlabel("relative time to cutoff point")
plt.ylabel("improvement over\nrandom sampling")
plt.legend()
plt.tight_layout()
plt.savefig("flowchart_plot_aggregated", dpi=300, bbox_inches="tight")
