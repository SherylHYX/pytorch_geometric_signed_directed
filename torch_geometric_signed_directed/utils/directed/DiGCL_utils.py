import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def drop_feature(x, drop_prob):
    r""" Drop feature funciton from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        x (PyTorch FloatTensor): Node features.
        drop_prob (float): Feature drop probability.
    Return types:
        x (PyTorch FloatTensor): Node features.
    """
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def pred_digcl(embeddings, y, train_index):
    r""" Generate predictions from embeddings from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        embeddings (PyTorch FloatTensor): Node embeddings.
        y (PyTorch LongTensor): Labels.
        train_index (NumPy array): Training index. 
    Return types:
        y_pred (NumPy array): Predicted labels.
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
    return y_pred