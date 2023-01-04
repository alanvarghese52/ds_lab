from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.40)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5);
plt.show()
x = pd.read_csv("cancer.csv")
a = np.array(x)
y = a[:, 30]
x = np.column_stack((x.malignant, x.benign))
x.shape
print(x), (y)
from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(x, y)
clf.predict([[120, 990]])
clf.predict([[85, 550]])
