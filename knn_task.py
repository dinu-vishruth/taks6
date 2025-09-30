# knn_task.py  -- small robustness fix added
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

def load_data():
    d = load_iris()
    X, y = d.data, d.target
    feature_names = d.feature_names
    return X, y, feature_names

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, y_train, y_val, scaler

def find_best_k(X_train, y_train, X_val, y_val, k_range=range(1,26)):
    accuracies = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, preds))
    return list(k_range), accuracies

def plot_accuracy_vs_k(k_vals, accs, savepath="results/accuracy_vs_k.png"):
    # Ensure directory exists for this specific save
    dirpath = os.path.dirname(savepath) or "."
    os.makedirs(dirpath, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(k_vals, accs, marker='o')
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("K vs Validation Accuracy")
    plt.grid(True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def train_final_and_report(X_train, y_train, X_val, y_val, best_k):
    clf = KNeighborsClassifier(n_neighbors=best_k)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)
    cr = classification_report(y_val, preds)
    return clf, acc, cm, cr, preds

def plot_decision_boundary_2D(X_2d, y, clf, title="Decision boundary", savepath="results/decision_boundary.png"):
    dirpath = os.path.dirname(savepath) or "."
    os.makedirs(dirpath, exist_ok=True)

    x_min, x_max = X_2d[:,0].min() - 1, X_2d[:,0].max() + 1
    y_min, y_max = X_2d[:,1].min() - 1, X_2d[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def main():
    # create results dir once for all saves (extra safety)
    os.makedirs("results", exist_ok=True)

    X, y, fnames = load_data()
    X_train_s, X_val_s, y_train, y_val, scaler = split_and_scale(X, y)

    k_vals, accs = find_best_k(X_train_s, y_train, X_val_s, y_val, range(1,26))
    print("k values:", k_vals)
    print("accuracies:", accs)
    plot_accuracy_vs_k(k_vals, accs)

    best_k = k_vals[np.argmax(accs)]
    print("Best k =", best_k)

    clf, acc, cm, cr, preds = train_final_and_report(X_train_s, y_train, X_val_s, y_val, best_k)
    print("Validation accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", cr)

    # Save model and scaler into results/
    joblib.dump({'model': clf, 'scaler': scaler}, os.path.join("results", "knn_model_and_scaler.joblib"))

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)

    knn_2d = KNeighborsClassifier(n_neighbors=best_k)
    knn_2d.fit(np.vstack((X_train_pca, X_val_pca)), np.hstack((y_train, y_val)))
    plot_decision_boundary_2D(np.vstack((X_train_pca, X_val_pca)), np.hstack((y_train, y_val)), knn_2d,
                              title=f"KNN decision boundary (k={best_k}) PCA-2D", savepath=os.path.join("results","decision_boundary_pca.png"))

if __name__ == "__main__":
    main()
