import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "hours_studied": [1,2,3,8,9,10,11],
    "attendance": [60,65,70,85,88,90,92]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "attendance"]]

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(X)

df["cluster"] = kmeans.labels_

print(df)


print("Centroids:\n", kmeans.cluster_centers_)
plt.scatter(X["hours_studied"], X["attendance"], c=df["cluster"])
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker="X",
    s=200
)
plt.xlabel("Hours Studied")
plt.ylabel("Attendance")
plt.show()