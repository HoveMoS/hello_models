# -*- coding: utf-8 -*-

# 使用ANN对鸢尾花分类


from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow.sklearn



mlflow.set_experiment("hello_wjf")


def eval_metrics(actual, pred):       ########### 定义 eval_metrics
    rmse = numpy.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def getData_1():

    iris = datasets.load_iris()
    X = iris.data   #样本特征矩阵，150*4矩阵，每行一个样本，每个样本维度是4
    y = iris.target #样本类别矩阵，150维行向量，每个元素代表一个样本的类别


    df1=pd.DataFrame(X, columns =['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
    df1['target']=y

    return df1

df=getData_1()


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:3],df['target'], test_size=0.3, random_state=42)
print(X_train, X_test, y_train, y_test)


#mlflow运行
with mlflow.start_run():


    model = LinearRegression()


    mlflow.log_param("fit_intercept", True)
    mlflow.log_param("normalize", False)

    model.fit(X_train,y_train)
    predict=model.predict(X_test)
    print(predict)

    #print(y_test.values)
    #print('神经网络分类:{:.3f}'.format(model.score(X_test, y_test)))


    (rmse, mae, r2) = eval_metrics(y_test, predict)


    #追踪三个性能参数
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)



