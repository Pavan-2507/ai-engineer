import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=load_breast_cancer()
X=data.data[:,:2]
y=data.target

print(X.shape)
print(y.shape)
print(data.target_names)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

linear_svm=SVC(
    kernel='linear',
    C=1.0,
    probability=False
)

linear_svm.fit(X_train_scaled,y_train)

# y_pred_linear=linear_svm.predict(X_test_scaled)

x_min,x_max=X_train_scaled[:,0].min()-1,X_train_scaled[:,0].max()+1
y_min,y_max=X_train_scaled[:,1].min()-1,X_train_scaled[:,1].max()+1

xx,yy=np.meshgrid(
    np.linspace(x_min,x_max,300),
    np.linspace(y_min,y_max,300)
)

Z=linear_svm.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=((7,5)))
plt.contourf(xx,yy,Z,alpha=0.3,cmap=cmap_light)
plt.scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=y_train,cmap=cmap_bold, edgecolor='k',
    s=40,
    label="Train data")

plt.xlabel("Sepal length (scaled)")
plt.ylabel("Sepal width (scaled)")
plt.title("SVM Decision Boundary (RBF kernel) on Iris (2 features)")
plt.show()


# Visualization mentally

# For each cell (i,j): ---> plt.contourf(xx,yy,Z,alpha=0.3,cmap=cmap_light)


# xx =
# [ [1, 2, 3],
#   [1, 2, 3],
#   [1, 2, 3] ]

# yy =
# [ [10,10,10],
#   [11,11,11],
#   [12,12,12] ]


# (i,j)
# (1,10)	0	red.         # Cell	Z value	Color
# (2,10)	0	red
# (3,10)	1	green
# (1,11)	0	red
# (2,11)	1	green
# (3,11)	1	green
# (1,12)	1	green
# (2,12)	1	green
# (3,12)	1	green

# Contourf shades these 9 areas with smooth color boundaries.