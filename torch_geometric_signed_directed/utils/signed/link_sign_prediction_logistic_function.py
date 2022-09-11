from typing import Tuple, Union

import numpy as np
from sklearn import linear_model, metrics


def link_sign_prediction_logistic_function(embeddings: np.ndarray, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, class_weight: Union[dict, str] = None) -> Tuple[float, float, float, float, float]:
    """
    link_sign_prediction_logistic_function [summary]
    Link sign prediction is a binary classification machine learning task. 
    It will return the metrics for link sign prediction (i.e., Accuracy, Binary-F1, Macro-F1, Micro-F1 and AUC).

    Args:
        embeddings (np.ndarray): The embeddings for signed graph.
        train_X (np.ndarray): The indices for training data (e.g., [[0, 1], [0, 2]])
        train_y (np.ndarray): The sign for training data (e.g., [[1, -1]])
        test_X (np.ndarray): The indices for test data (e.g., [[1, 2]])
        test_y (np.ndarray): The sign for test data (e.g., [[1]])
        class_weight (Union[dict, str], optional): Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.. Defaults to None.

    Returns:
        [type]: The metrics for link sign prediction task.
        Tuple[float,float,float,float,float]: Accuracy, Binary-F1, Macro-F1, Micro-F1 and AUC.

    """
    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    logistic_function = linear_model.LogisticRegression(
        solver='lbfgs', max_iter=1000)
    logistic_function.fit(train_X1, train_y)
    pred = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)
    accuracy = metrics.accuracy_score(test_y, pred)
    f1 = metrics.f1_score(test_y, pred, average='binary')
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    auc_score = metrics.roc_auc_score(test_y, pred_p[:, 1])
    return accuracy, f1, f1_macro, f1_micro, auc_score
