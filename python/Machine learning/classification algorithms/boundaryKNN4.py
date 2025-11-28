import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


iris=load_iris()

X=iris.data[:,:2]
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#scalling

scaler=StandardScaler()

X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)

k=5
model=KNeighborsClassifier(
    n_neighbors=k
)

model.fit(X_train_scale,y_train)

x_min,x_max=X_train_scale[:,0].min()-1,X_train_scale[:,0].max()+1
y_min,y_max=X_train_scale[:,1].min()-1,X_train_scale[:,1].max()+1

xx,yy=np.meshgrid(
    np.linspace(x_min,x_max,300),
    np.linspace(y_min,y_max,300)
)

Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


plt.figure(figsize=(7, 5))
plt.contourf(xx,yy,Z, alpha=0.3,cmap=cmap_light)
plt.scatter(X_train_scale[:,0],X_train_scale[:,1],c=y_train,cmap=cmap_bold,edgecolors='k',s=40)
plt.xlabel("Sepal length (scaled)")
plt.ylabel("Sepal width (scaled)")
plt.title("KNN Decision Boundary (k=5) using 2 features")
plt.show()

