import numpy as np

# Input path
path = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (4, 3)]

# Convert to a NumPy array
P = np.array(path)

# Compute differences between consecutive points
Pp = np.diff(P, axis=0)

# Compute delta (differences of Pp)
delta = np.diff(Pp, axis=0)

# Identify indices where delta equals 0 (indicating collinearity)
indices_to_remove = np.where((delta == 0).all(axis=1))[0] + 1

# Remove the intermediate points
P2 = np.delete(P, indices_to_remove, axis=0)

# Output the simplified path
print("Original Path:", P)
print("Simplified Path:", P2) 