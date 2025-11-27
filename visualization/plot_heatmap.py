import numpy as np
import matplotlib.pyplot as plt

# Load CSV file (MPI result)
data = np.loadtxt("u_mpi_final.csv", delimiter=",")

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap="hot", origin="lower")
plt.colorbar(label="Temperature")
plt.title("Heatmap of Final Temperature Field (MPI Solver)")
plt.xlabel("Y index")
plt.ylabel("X index")

plt.savefig("heatmap.png", dpi=300)
plt.show()

