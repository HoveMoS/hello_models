# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report
from common.utils import plot_learning_curve
from sklearn.model_selection import ShuffleSplit

import mlflow

# TODO 忽略警告
import warnings
#warnings.warn("deprecated", DeprecationWarning)
#warnings.filterwarnings("ignore")



mlflow.set_experiment("logistic for classifier")

mlflow.sklearn.autolog()

#mlflow.run()

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
#     X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
# print(cancer.data[0])
#
# cancer.feature_names
# print(cancer.feature_names)  # zzj





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#mlflow运行

#with mlflow.start_run(run_name='test',tags={"priority": "P2"}):
with mlflow.start_run(run_name='test'):

    # 模型训练
    model = LogisticRegression()
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




    title = "Learning Curves (LogisticRegression)"
    model2=LogisticRegression()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #estimator = SVC(gamma=0.001)
    a,fig=plot_learning_curve(model2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)


    # TODO 记录图片
    #plt.show()
    mlflow.log_figure(fig, "learning_curve.png")





# ROC,AUC

# from sklearn.metrics import roc_curve, auc

# 方案 A
#
#
# # 为每个类别计算ROC曲线和AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(2):
#     fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # fpr[0].shape==tpr[0].shape==(21, ), fpr[1].shape==tpr[1].shape==(35, ), fpr[2].shape==tpr[2].shape==(33, )
# # roc_auc {0: 0.9118165784832452, 1: 0.6029629629629629, 2: 0.7859477124183007}
#
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()







# 方案 B


# fpr, tpr, thersholds = roc_curve(y_true, y_pred, pos_label=2)
#
# for i, value in enumerate(thersholds):
#     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
#
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()













