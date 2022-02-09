

import numpy as np
from sklearn import linear_model, metrics

def link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y):
   
    train_X1 = []
    test_X1  = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))


    logistic_function = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_function.fit(train_X1, train_y)
    pred   = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)

    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    accuracy  =  metrics.accuracy_score(test_y, pred)
    f1        =  metrics.f1_score(test_y, pred)
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])

    return accuracy, f1, f1_macro, f1_micro, auc_score