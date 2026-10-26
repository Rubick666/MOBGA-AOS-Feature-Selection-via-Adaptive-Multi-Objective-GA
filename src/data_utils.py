import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_dataset(path):
    data = pd.read_csv(path, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # ---- FIX: force labels to strings, then encode ----
    y = y.astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # ---- normalize features for kNN ----
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

def knn_error(X, y, mask):
    if np.sum(mask) == 0:
        return 1.0

    X_sel = X[:, mask == 1]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    errors = []

    for train, test in kf.split(X_sel):
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_sel[train], y[train])
        acc = clf.score(X_sel[test], y[test])
        errors.append(1 - acc)

    return np.mean(errors)