import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


def printing_Kfold_scores(x_train_data, y_train_data):
    # 将训练集切分成5份，做交叉验证
    kf = KFold(n_splits=5, shuffle=False)
    kf.get_n_splits(x_train_data)

    # 正则化惩罚项系数
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        # 循环进行交叉验证
        for iteration, indices in kf.split(x_train_data):
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            lr.fit(x_train_data.iloc[iteration, :], y_train_data.iloc[iteration, :].values.ravel())

            y_pred_undersample = lr.predict(x_train_data.iloc[indices, :].values)

            recall_acc = recall_score(y_train_data.iloc[indices, :].values, y_pred_undersample)  # 计算召回率
            recall_accs.append(recall_acc)

            print('recall score = ', recall_acc)

        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
    results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c


data = pd.read_csv("./data/creditcard.csv")
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

# 取其index
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                    , y_undersample
                                                                                                    , test_size=0.3
                                                                                                    , random_state=0)
best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

