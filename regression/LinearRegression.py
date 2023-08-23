# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report
from common.utils import plot_learning_curve
from sklearn.model_selection import ShuffleSplit

import mlflow



mlflow.set_experiment("linear for classifier")

mlflow.sklearn.autolog()

#mlflow.run()

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#mlflow运行

#with mlflow.start_run(run_name='test',tags={"priority": "P2"}):
with mlflow.start_run(run_name='test'):

    # 模型训练
    model = LinearRegression()
    #mlflow.log_param("test", 2)

    model.fit(X_train, y_train)


    # 样本预测
    y_pred = model.predict(X_test)
    print('matchs: {0}/{1}'.format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))



    y_true = y_test

    print(confusion_matrix(y_true, y_pred))

    target_names = ['class 0', 'class 1']
    print(classification_report(y_true, y_pred, target_names=target_names))


    print('accuracy_score')
    res_accuracy_score=accuracy_score(y_true, y_pred)
    print(res_accuracy_score)
    # print(accuracy_score(y_true, y_pred, normalize=False))

    print('precision_score')
    res_precision_score = precision_score(y_true, y_pred, average='macro')
    print(res_precision_score)
    print(precision_score(y_true, y_pred, average='micro'))
    print(precision_score(y_true, y_pred, average='weighted'))
    print(precision_score(y_true, y_pred, average=None))

    print('recall_score')
    res_recall_score = recall_score(y_true, y_pred, average='macro')
    print(res_accuracy_score)
    print(recall_score(y_true, y_pred, average='micro'))
    print(recall_score(y_true, y_pred, average='weighted'))
    print(recall_score(y_true, y_pred, average=None))

    print('f1_score')
    res_f1_score = f1_score(y_true, y_pred, average='macro')
    print(res_f1_score)
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='weighted'))
    print(f1_score(y_true, y_pred, average=None))

    mlflow.log_metric("accuracy_score", res_accuracy_score)
    mlflow.log_metric("precision_score", res_precision_score)
    mlflow.log_metric("recall_score", res_recall_score)
    mlflow.log_metric("f1_score", res_f1_score)




    title = "Learning Curves (LinearRegression)"
    model2=LinearRegression()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #estimator = SVC(gamma=0.001)
    a,fig=plot_learning_curve(model2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)


    #plt.show()
    mlflow.log_figure(fig, "learning_curve.png")



