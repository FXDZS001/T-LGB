import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

import xgboost
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings("ignore") 


def heat_map():
    data = pd.read_excel(r'C:\Users\admin\Desktop\论文原始数据\2018白沙.xlsx', sheet_name='Sheet1')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] =False 


    
    data = pd.DataFrame(data)
    corr = data.corr(method='pearson')
    print(corr)
    
    sns.heatmap(corr,
                annot=True,  
                center=0.5,  
                fmt='.2f',  
                linewidth=0.5,  
                linecolor='blue',  
                vmin=0, vmax=1,  
                xticklabels=True, yticklabels=True,  
                square=True,  
                cbar=True,  

                )
    plt.savefig("baishatu.png", dpi=600)
    plt.ion() 

def prep_clf(obs,pre, threshold=0.1):

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



    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits/(hits + falsealarms + misses)

def FAR(obs, pre, threshold=0.1):
    '''
    func: falsealarms / (hits + falsealarms)
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
    func :  misses / (hits + misses)
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
            model.fit(train_featrues, train_labels.astype('int'))  

            plt.figure(figsize=(12, 6))
            lgb.plot_importance(model, xlabel='F score', max_num_features=30)
            plt.title("Featurertances")
            plt.show()

def get_train_x_y(sample,label):
    x = sample
    y = label
    x = StandardScaler().fit_transform(x)
    return x, y

import os, numpy as np, pandas as pd

def save_metrics_and_probs(model_name, save_root, iteration, 
                           y_test, y_pred, y_prob, metrics_dict):

    model_dir = os.path.join(save_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(model_dir, f"y_true_iter{iteration}.npy"), y_test.astype(int))
    np.save(os.path.join(model_dir, f"y_prob_iter{iteration}.npy"), y_prob.astype(float))


    metrics_dict['Iteration'].append(iteration)
    metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
    metrics_dict['Recall'].append(recall_score(y_test, y_pred) * 100)
    metrics_dict['F1_Score'].append(f1_score(y_test, y_pred) * 100)
    metrics_dict['TS_Score'].append(TS(y_test, y_pred) * 100)
    metrics_dict['FAR'].append(FAR(y_test, y_pred) * 100)
    metrics_dict['MAR'].append(MAR(y_test, y_pred) * 100)

    return model_dir  


def finalize_metrics(model_name, model_dir, metrics_dict):
    """在循环结束后保存该模型的CSV"""
    df = pd.DataFrame(metrics_dict)
    out_csv = os.path.join(model_dir, f"metrics_{model_name}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {model_name} 指标CSV已保存：{out_csv}")

def func_C():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = LGBMClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("T-LGB", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("T-LGB", model_dir, metrics)


def GBDT():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = GradientBoostingClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("GBDT", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("GBDT", model_dir, metrics)


def XGBoost():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = XGBClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("XGBoost", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("XGBoost", model_dir, metrics)


def RandomForest():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = RandomForestClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("RandomForest", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("RandomForest", model_dir, metrics)


def AdaBoost():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = AdaBoostClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("AdaBoost", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("AdaBoost", model_dir, metrics)





if __name__ == "__main__":


    # importance_feature()
    # data()
    # func_C()
    # func_R()
    # func_XGB()
    # func_GBDT()
    # func_RandomForest()
    # func_AdaBoost()
    # GBDT()
    # XGBoost()
    # RandomForest()
    # AdaBoost()

