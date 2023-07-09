from sklearn.metrics import classification_report

x_true = [0, 1, 2, 2, 2]
y_prec = [0, 0, 2, 2, 1]
print(classification_report(x_true,y_prec))
# precision 精确率
# recall 召回率
# f1
""""
precision    recall  f1-score   support

           0       0.50      1.00      0.67         1
           1       0.00      0.00      0.00         1
           2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
"""