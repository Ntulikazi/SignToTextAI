import numpy as np

data = np.loadtxt("hand_detect.csv", delimiter=",", dtype=str)

print("Dataset loaded successfully")
print("Shape:", data.shape)

# Split into features and labels
X = data[:, :-1].astype(float)  # first 63 columns
y = data[:, -1]                 # last column

unique, counts = np.unique(y, return_counts=True)

print("\nSamples per label:")
for label, count in zip(unique, counts):
    print(label, ":", count)

print("Ntuli")
print(X.shape)
print(y.shape)
