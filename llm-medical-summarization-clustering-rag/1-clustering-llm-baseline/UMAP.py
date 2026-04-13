import numpy as np
import pandas as pd
import umap
import sklearn.cluster as cluster
from Clustering_evaluation import Accuracy, NMI, SC, DB
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler

patient_dataset = pd.read_csv("patient_dataset_with_reports_2.csv")
patient_dataset2 = pd.read_csv("patient_dataset_with_reports.csv")

d = patient_dataset.drop(columns=["Hypertension_Stage", "Clinical_Report"])
y = patient_dataset["Hypertension_Stage"]  # Using 'condition label' as the target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(d)

print(np.shape(d))
print(np.shape(y))

UMAP=umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0)

#for i in range(1):
time_start = time.time()
embedding = UMAP.fit_transform(X_scaled)
end_time = time.time() - time_start

df = pd.DataFrame(embedding)
df.to_csv("UMAP-embedding-NC2.csv", index=False)

kmeans = cluster.KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(embedding)

patient_dataset2["UMAP_kmeans_labels"]=y_pred
patient_dataset2.to_csv("patient_dataset_with_reports.csv", index=False)

print("UMAP results")
print("Time execution= ",end_time)
print("Accuracy score= ", Accuracy(y, y_pred))
print("NMI score= ", NMI(y, y_pred))
print("silhouette score= ", SC(embedding, y_pred))
print("DBI score= ", DB(embedding, y_pred))

colors={0: 'lime', 1: 'cyan', 2: 'purple'}
label_name= {0: 'High Risk Group', 1: 'Average Risk Group', 2: 'Normal Risk Group'}
x = embedding[:, 0]
y = embedding[:, 1]
df['target'] = y_pred
y_pred=df['target']
plt.figure(figsize=(8, 5))
for label in y_pred.unique():
    plt.scatter(x[y_pred==label], y[y_pred==label], c=colors[label],label=label_name[label], alpha=0.7)

"""df['target'] = y
df['x'] = embedding[:, 0]
df['y'] = embedding[:, 1]
plt.figure(figsize=(6, 4))
sns.scatterplot(x='x', y='y', hue='target', palette=sns.color_palette("hsv", 3), data=df)"""
plt.legend(title="Target", fontsize=12)
#plt.title("UMAP Visualization")
plt.xlabel("UMAP First Component", fontsize=12)
plt.ylabel("UMAP Second Component", fontsize=12)
plt.show()
