# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:45:02 2024

@author: ByteBard8
"""
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

############
#QUESTION 1#
############
def get_unique_digits(X):
    unique_digits = set()
    X_unique_digits = []
    for i in range(len(y)):
        if y[i] not in unique_digits:
            X_unique_digits.append(X[i])  # Add the feature (image) to the list
            unique_digits.add(y[i])  # Add the digit to the set of found targets
        if len(unique_digits) == 10:  # We need one instance for each digit 0-9
            break
    return X_unique_digits


def plot_digits(X_unique_digits,reshape_size):
    fig, axes = plt.subplots(2, 5)
    axes = axes.ravel()
    for i in range(10):
        if i < len(X_unique_digits):
            axes[i].imshow(X_unique_digits[i].reshape(reshape_size), cmap=plt.cm.Blues)
            axes[i].axis('off')  # Turn off axis
        else:
            axes[i].axis('off')  # Keep remaining cells blank
    # Display the plot
    plt.tight_layout()
    plt.show()


#Retrieve and load the mnist_784 dataset of 70,000 instances
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#set features and target
X, y = mnist['data'], mnist['target']
print(X.shape)
print(y.shape)
print(f"Target digits:\n{','.join(np.unique(y))}")

X_unique_digits = get_unique_digits(X)

plot_digits(X_unique_digits,reshape_size=(28,28))

#Use PCA to retrieve the and principal component and output their explained variance ratio. [5 points]
pca = PCA(n_components=2)  # We only need the first 2 components
x_train_pca = pca.fit_transform(X)

# Output the explained variance ratio for the first two components
explained_variance_ratio = pca.explained_variance_ratio_
y_train = y
print(f"Explained variance ratio for the first component: {explained_variance_ratio[0]:.4f}")
print(f"Explained variance ratio for the second component: {explained_variance_ratio[1]:.4f}")
#Plot the projections of the and principal component onto a 2D hyperplane. [5 points]
for i in range(10):
    plt.scatter(x_train_pca[y_train == str(i), 0], x_train_pca[y_train == str(i), 1], label=str(i), marker=".")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend()
plt.title("MNIST Dataset after PCA")
plt.show()
#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]

dimensions=154
ipca = IncrementalPCA(n_components=dimensions)
X_reduced = ipca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

#Display the original and compressed digits from (5). [5 points]
X_reduced_unique_digits = get_unique_digits(X_reduced)

plot_digits(X_reduced_unique_digits,reshape_size=(11,14))
#Create a video discussing the code and result for each question. Discuss challenges you confronted and solutions to overcoming them, if applicable [15 points]


############
#QUESTION 2#
############
# Generate Swiss roll dataset. [5 points]
from sklearn.datasets import make_swiss_roll
# Generate the Swiss Roll dataset
n_samples = 1000  # Number of points
X, y = make_swiss_roll(n_samples=n_samples)
# Plot the resulting generated Swiss roll dataset. [2 points]
plt.scatter(X[:,0], X[:,1], cmap=plt.cm.Blues, c=y)
plt.show()
# Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points). [6 points]
from sklearn.decomposition import KernelPCA

# Apply Kernel PCA with a linear kernel (2 principal components)
kpca_linear = KernelPCA(n_components=2, kernel='linear')
X_kpca_linear = kpca_linear.fit_transform(X)

# Apply Kernel PCA with an RBF (Gaussian) kernel (2 principal components)
kpca_rbf = KernelPCA(n_components=2, kernel='rbf')
X_kpca_rbf = kpca_rbf.fit_transform(X)

# Apply Kernel PCA with a Sigmoid kernel (2 principal components)
kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)
# Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3). Explain and compare the results [6 points]
# Define function to plot results
def plot_kpca(X_kpca, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


plot_kpca(X_kpca_linear, "Kernel PCA with Linear Kernel")
plot_kpca(X_kpca_rbf, "Kernel PCA with RBF Kernel")
plot_kpca(X_kpca_sigmoid, "Kernel PCA with Sigmoid Kernel")
# Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV. [14 points]
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

threshold_value = np.average(y)
y_discrete = np.where(y > threshold_value, 1, 0)

strat_kfold = StratifiedKFold(shuffle=True, random_state=62)
for train_index, test_index in strat_kfold.split(X, y_discrete):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_discrete[train_index], y_discrete[test_index]
    break

clf = Pipeline([
  ("kpca", KernelPCA(n_components=2)),
   ("log_reg", LogisticRegression())
])

param_grid = [{
  "kpca__gamma": np.linspace(0.03, 0.05, 10),
  "kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best model={best_model}")
print(f"Parameters={grid_search.best_params_}")

# Plot the results from using GridSearchCV in (5). [2 points]
X_kpca = best_model["kpca"].transform(X)

plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.Blues)
plt.show()
# Create a video discussing the code and result for each question. Discuss challenges you confronted and solutions to overcoming them, if applicable [15 points]