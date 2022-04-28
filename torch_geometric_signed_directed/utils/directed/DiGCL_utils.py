from typing import List

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def drop_feature(x: torch.FloatTensor, drop_prob: float):
    r""" Drop feature funciton from the
    `Directed Graph Contrastive Learning 
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Arg types:
        * **x** (PyTorch FloatTensor) - Node features.
        * **drop_prob** (float) - Feature drop probability.

    Return types:
        * **x** (PyTorch FloatTensor) - Node features.
    """
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def pred_digcl_node(embeddings: torch.FloatTensor, y: torch.LongTensor, train_index: List[int], test_index: List[int] = None):
    r""" Generate predictions from embeddings from the
    `Directed Graph Contrastive Learning
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Arg types:
        * **embeddings** (PyTorch FloatTensor) - Node embeddings.
        * **y** (PyTorch LongTensor) - Labels.
        * **train_index** (NumPy array) - Training index. 
        * **test_index** (NumPy array) - Testing index. 

    Return types:
        * **y_pred** (NumPy array) - Predicted labels.
    """
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')
    X_train = X[train_index]
    y_train = Y[train_index]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = np.argmax(clf.predict(X), axis=1)

    if test_index is None:
        return y_pred
    else:
        return y_pred[test_index]


def pred_digcl_link(embeddings: torch.FloatTensor, y: torch.LongTensor, train_index: List[int], test_index: List[int]):
    r""" Generate predictions from embeddings from the
    `Directed Graph Contrastive Learning
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Arg types:
        * **embeddings** (PyTorch FloatTensor) - Node embeddings.
        * **y** (PyTorch LongTensor) - Labels.
        * **train_index** (NumPy array) - Training index. 
        * **test_index** (NumPy array) - Testing index. 

    Return types:
        * **y_pred** (NumPy array) - Predicted labels.
    """
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')
    X_train1 = X[train_index[:, 0]]
    X_train2 = X[train_index[:, 1]]
    X_train = np.c_[X_train1, X_train2]
    y_train = Y

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    X_test1 = X[test_index[:, 0]]
    X_test2 = X[test_index[:, 1]]
    X_test = np.c_[X_test1, X_test2]
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    return y_pred
