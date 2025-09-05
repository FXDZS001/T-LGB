import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

import xgboost
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore") #过滤掉警告的意思



def heat_map():
    data = pd.read_excel(r'C:\Users\admin\Desktop\论文原始数据\2018白沙.xlsx', sheet_name='Sheet1')
    #图片显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] =False #减号unicode编码


    #计算各变量之间的相关系数
    data = pd.DataFrame(data)
    corr = data.corr(method='pearson')
    print(corr)
    #开始绘图
    sns.heatmap(corr,
                annot=True,  # 显示相关系数的数据
                center=0.5,  # 居中
                fmt='.2f',  # 只显示两位小数
                linewidth=0.5,  # 设置每个单元格的距离
                linecolor='blue',  # 设置间距线的颜色
                vmin=0, vmax=1,  # 设置数值最小值和最大值
                xticklabels=True, yticklabels=True,  # 显示x轴和y轴
                square=True,  # 每个方格都是正方形
                cbar=True,  # 绘制颜色条
                cmap='coolwarm_r',  # 设置热力图颜色
                )
    plt.savefig("我是废强热力图.png", dpi=600)#保存图片，分辨率为600
    plt.ion() #显示图片

def prep_clf(obs,pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    #根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives

def TS(obs, pre, threshold=0.1):

    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses)
          alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits/(hits + falsealarms + misses)

def FAR(obs, pre, threshold=0.1):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms)
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)

def MAR(obs, pre, threshold=0.1):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return misses / (hits + misses)

def importance_feature():
    root_path = r'F:\毕业'
    stations = ['总站']
    for i in range(len(stations)):
        for j in range(2, 3):
            # read_ec = os.path.join(root_path, stations[i], '分类', '低', 'high_ec', 'feature' + str(j) + '.npy')
            read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
            read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
            read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
            sc = np.load(read_sc, allow_pickle=True)
            ec_low = np.load(read_eclow, allow_pickle=True)
            label = np.load(read_label, allow_pickle=True)
            label = label.squeeze()
            ec_low = ec_low.squeeze()
            sample = np.concatenate((sc, ec_low), axis=1)
            # sample = ec
            train_featrues = sample
            train_labels = label
            model = LGBMClassifier(objective='regression',
                                   max_depth=8,
                                   learning_rate=0.1, n_estimators=7850,
                                   bagging_fraction=0.85, feature_fraction=0.85)
            model.fit(train_featrues, train_labels.astype('int'))  # 无法识别object类型，需强制转换成int类型

            plt.figure(figsize=(12, 6))
            lgb.plot_importance(model, xlabel='F score', max_num_features=30)
            plt.title("Featurertances")
            plt.show()

def data():
    root_path = r'F:\毕业'
    for j in range(21, 22):
        read_ec = os.path.join(root_path, '数据', 'TiVec_samples' + str(j) + '.npy')
        ec = np.load(read_ec, allow_pickle=True)
        high = ec[:, :, 4, 4]
        high_low = high.squeeze()
        np.save(os.path.join(root_path, '总站', '分类', '低', '其他算法', 'high_low' + str(j) + '.npy'), high_low)
# 标准化
def get_train_x_y(sample,label):
    x = sample
    y = label
    x = StandardScaler().fit_transform(x)
    return x, y
# 分类
def func_C():
    root_path = r'D:\hainan'
    for j in range(1, 22):
        read_ec = os.path.join(root_path, 'Balanced', 'feature', 'feature' + str(j) + '.npy')
        read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
        read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
        sc = np.load(read_sc, allow_pickle=True)
        ec = np.load(read_ec, allow_pickle=True)
        ec_low = np.load(read_eclow, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        label = label.squeeze()
        ec_low = ec_low.squeeze()
        sample = np.concatenate((sc, ec_low, ec), axis=1)
        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        model = LGBMClassifier()
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ts = TS(y_test, y_pred, threshold=0.1)
        # ets = ETS(test_labels, y_pred, threshold=0.1)
        far = FAR(y_test, y_pred, threshold=0.1)
        mar = MAR(y_test, y_pred, threshold=0.1)
        # pod = POD(test_labels, y_pred, threshold=0.1)
        # bias = BIAS(test_labels, y_pred, threshold=0.1)
        # hss = HSS(test_labels, y_pred, threshold=0.1)
        print(j)
        print('%.2f' % (acc * 100))
        # print('%.2f' % (precision * 100))
        print('%.2f' % (recall * 100))
        print('%.2f' % (ts * 100))
        print('%.2f' % (mar * 100))
        print('%.2f' % (far * 100))
        # print('%.2f' % (f1 * 100))
        # print('%.2f' % (ets * 100))
        # print('%.2f' % (pod * 100))
        # print('%.2f' % (bias * 100))
        # print('%.2f' % (hss * 100))

# 横向对比
def func_XGB():
    root_path = r'D:\hainan\回归'
    for j in range(1, 22):
        read_samples = os.path.join(root_path, 'samples' , 'samples' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'label', 'samples' + str(j) + '.npy')
        samples = np.load(read_samples, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        # label = label.squeeze()
        # ec_low = ec_low.squeeze()
        x_train, y_train = get_train_x_y(samples, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
            # 添加其他你想要调整的参数
        }
        model = XGBRegressor()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        best_params = model.best_params_
        print(f"最佳参数组合: {best_params}")
        # best_model = LGBMRegressor(**best_params)
        # best_model.fit(x_train, y_train.astype('int'))


        y_pred = model.predict(x_test)
        MEA = mean_absolute_error(y_test, y_pred)
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        print('MEA:',MEA)
        print('RMSE:',RMSE)

def func_R():
    root_path = r'D:\hainan'
    for j in range(21,22):
        read_samples = os.path.join(root_path, 'samples',  'samples' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'label', 'samples' + str(j) + '.npy')
        samples = np.load(read_samples, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        # label = label.squeeze()
        # ec_low = ec_low.squeeze()
        x_train, y_train = get_train_x_y(samples, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.015, 0.5],
            'reg_lambda' : [0, 0.1, 0.5, 1],
            'max_depth': [3, 5, 7]
            # 添加其他你想要调整的参数
        }
        model = LGBMRegressor()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        best_params = model.best_params_
        best_model = LGBMRegressor(**best_params)
        best_model.fit(x_train, y_train.astype('int'))


        y_pred = best_model.predict(x_test)
        MAE = mean_absolute_error(y_test, y_pred)
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        print('j:',j)
        print(f"最佳参数组合: {best_params}")
        print('MAE:',MAE)
        print('RMSE:',RMSE)
        print("-" * 20)  # 打印分隔线
def func_GBDT():
    root_path = r'D:\hainan\回归'
    for j in range(1, 22):
        read_samples = os.path.join(root_path, 'samples', 'samples' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'label', 'samples' + str(j) + '.npy')
        samples = np.load(read_samples, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        x_train, y_train = get_train_x_y(samples, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.015, 0.5],
            'max_depth': [3, 5, 7]
            # 添加其他你想要调整的参数
        }
        model = GradientBoostingRegressor()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        best_params = model.best_params_
        # best_model = LGBMRegressor(**best_params)
        # best_model.fit(x_train, y_train.astype('int'))
        print(f"最佳参数组合: {best_params}")
        y_pred = model.predict(x_test)
        MAE = mean_absolute_error(y_test, y_pred)
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        print('MAE:', MAE)
        print('RMSE:', RMSE)

def func_RandomForest():
    root_path = r'D:\hainan\回归'
    for j in range(1, 22):
        read_samples = os.path.join(root_path, 'samples', 'samples' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'label', 'samples' + str(j) + '.npy')
        samples = np.load(read_samples, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        # label = label.squeeze()
        # ec_low = ec_low.squeeze()
        x_train, y_train = get_train_x_y(samples, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'min_samples_split': [2, 5, 10],
            'max_depth': [3, 5, 7],
            # 添加其他你想要调整的参数
        }
        model = RandomForestRegressor()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        best_params = model.best_params_
        print(f"随机森林最佳参数组合: {best_params}")
        # best_model = LGBMRegressor(**best_params)
        # best_model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        MEA = mean_absolute_error(y_test, y_pred)
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        print('MEA:',MEA)
        print('RMSE:',RMSE)


def func_AdaBoost():
    root_path = r'D:\hainan\回归'
    for j in range(1, 22):
        read_samples = os.path.join(root_path, 'samples', 'samples' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'label', 'samples' + str(j) + '.npy')
        samples = np.load(read_samples, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        # label = label.squeeze()
        # ec_low = ec_low.squeeze()
        x_train, y_train = get_train_x_y(samples, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
             'loss': ['linear', 'square', 'exponential'],
            # 添加其他你想要调整的参数
        }
        model = AdaBoostRegressor()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        best_params = model.best_params_
        print(f"Adaboost最佳参数组合: {best_params}")
        # best_model = LGBMRegressor(**best_params)
        # best_model.fit(x_train, y_train.astype('int'))

        y_pred = model.predict(x_test)
        MEA = mean_absolute_error(y_test, y_pred)
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        print('MEA:', MEA)
        print('RMSE:', RMSE)


def GBDT():
    root_path = r'F:\毕业'
    for j in range(1, 22):
        read_ec = os.path.join(root_path, 'Balanced', '其他算法', 'ec_high' + str(j) + '.npy')
        read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
        read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
        sc = np.load(read_sc, allow_pickle=True)
        ec = np.load(read_ec, allow_pickle=True)
        ec_low = np.load(read_eclow, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        label = label.squeeze()
        ec_low = ec_low.squeeze()
        sample = np.concatenate((sc, ec_low, ec), axis=1)
        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('%.2f' % (acc * 100))

def XGBoost():
    root_path = r'F:\毕业'
    for j in range(1, 22):
        read_ec = os.path.join(root_path, 'Balanced', '其他算法', 'ec_high' + str(j) + '.npy')
        read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
        read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
        sc = np.load(read_sc, allow_pickle=True)
        ec = np.load(read_ec, allow_pickle=True)
        ec_low = np.load(read_eclow, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        label = label.squeeze()
        ec_low = ec_low.squeeze()
        sample = np.concatenate((sc, ec_low, ec), axis=1)
        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        model = XGBClassifier()
        model.fit( x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('%.2f' % (acc * 100))

def RandomForest():
    root_path = r'F:\毕业'
    for j in range(1, 22):
        read_ec = os.path.join(root_path, 'Balanced', '其他算法', 'ec_high' + str(j) + '.npy')
        read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
        read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
        sc = np.load(read_sc, allow_pickle=True)
        ec = np.load(read_ec, allow_pickle=True)
        ec_low = np.load(read_eclow, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        label = label.squeeze()
        ec_low = ec_low.squeeze()
        sample = np.concatenate((sc, ec_low, ec), axis=1)
        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        model = RandomForestClassifier()
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('%.2f' % (acc * 100))

def AdaBoost():
    root_path = r'F:\毕业'
    for j in range(1, 22):
        read_ec = os.path.join(root_path, 'Balanced', '其他算法', 'ec_high' + str(j) + '.npy')
        read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
        read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
        sc = np.load(read_sc, allow_pickle=True)
        ec = np.load(read_ec, allow_pickle=True)
        ec_low = np.load(read_eclow, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True)
        label = label.squeeze()
        ec_low = ec_low.squeeze()
        sample = np.concatenate((sc, ec_low, ec), axis=1)
        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        model = AdaBoostClassifier()
        model.fit(x_train, y_train.astype('int'))  # 无法识别object类型，需强制转换成int类型
        # joblib.dump(model, f'./models/{stations[i]}/class{i+1}.pkl')  # 保存模型
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('%.2f' % (acc * 100))


if __name__ == "__main__":


    # importance_feature()
    # data()
    # func_C()
    # func_R()
    # func_XGB()
    # func_GBDT()
    func_RandomForest()
    func_AdaBoost()
    # GBDT()
    # XGBoost()
    # RandomForest()
    # AdaBoost()


