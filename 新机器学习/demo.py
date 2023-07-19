import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import cross_val_predict
import itertools

# 读入训练集
data = pd.read_csv("./creditcard.csv")
data['Class'].value_counts()
data.isnull().all()

# 数据标准化处理
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
data.head()

# 数据标签直方图
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes.plot(kind='bar')

plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# 定义交叉验证的函数
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)
    c_param_range = [0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range)), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    j = 0

    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正则化惩罚力度: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []

        for iteration, indices in enumerate(fold.split(x_train_data)):
            lr = LogisticRegression(C=c_param, penalty='l2', max_iter=3000)
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': 召回率 = ', recall_acc)

        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率 ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']

    # 打印最好的结果
    print('********************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('********************************************')

    return best_c


# 定义混淆矩阵

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 原数据集方案
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("原始训练集包含样本数量: ", len(X_train))
print("原始测试集包含样本数量: ", len(X_test))
print("原始样本总数: ", len(X_train) + len(X_test))

best_c = printing_Kfold_scores(X_train, y_train)
lr = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
lr.fit(X_train, y_train.values.ravel())
y_pred = lr.predict(X_test.values)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print('召回率:', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion matrix")
plt.show()

# 下采样方案
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = data.iloc[under_sample_indices, :]
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

print("正常样本所占整体比例: ", len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print("异常样本所占整体比例: ", len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print("下采样策略总体样本数量: ", len(under_sample_data))

# 下采样数据集进行划分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)
print("下采样训练集包含样本数量: ", len(X_train_undersample))
print("下采样测试集包含样本数量: ", len(X_test_undersample))
print("下采样样本总数: ", len(X_train_undersample) + len(X_test_undersample))

# 交叉验证，找到最好的参数
best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

# 绘制在下采样的训练集中的混淆矩阵
lr = LogisticRegression(C=best_c, penalty='l2', max_iter=3000)
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print("召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

# 验证下采样方案在原始数据集中的结果
lr = LogisticRegression(C=best_c, penalty='l2', max_iter=3000)
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print("召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

# 上采样方案
credit_cards = pd.read_csv('creditcard.csv')
columns = credit_cards.columns
features_columns = columns.delete(len(columns) - 1)
features = credit_cards[features_columns]
labels = credit_cards['Class']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=0)

oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_resample(features_train, labels_train)
print(len(labels_train[labels_train == 0]))
print(len(labels_train[labels_train == 1]))
print(len(os_labels[os_labels == 1]))
print(len(os_labels))

os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)

best_c = printing_Kfold_scores(os_features, os_labels)

lr = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
lr.fit(os_features, os_labels.values.ravel())
y_pred = lr.predict(features_test.values)
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)
print('召回率:', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion matrix")
plt.show()

