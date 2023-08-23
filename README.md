

重要改动记录

common.utils.plot_learning_curve
为原版绘制曲线修改而来
修改内容：
原函数返回plt对象，
新函数新增第二个返回值-fig对象,故新版返回 plt,fig




model1和model2为同一机器学习模型的未拟合模型
评估指标采用model1（未采用交叉分离）计算
学习曲线采用model2（采用交叉分离）计算




TODO

lightgbm需要添加学习曲线绘制

待添加模型 SVM

