import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR

data = pd.read_csv("msft_stockprices_dataset.csv", sep=",")

data.tail()

data['date_number'] = [i for i in range(1,len(data['Date'])+1)]

train_data, test_data = train_test_split(data, test_size=0.2)

ErrorTolerance = 0.01

features = ["date_number", "High Price", "Low Price", "Open Price", "Volume"]
# features = ['date_number']

target_field = 'Close Price'

X_train = np.column_stack(train_data[fn] for fn in features)
y_train = train_data[target_field]

test_data = test_data.sort_values(['date_number'], ascending=[1])

X_test = np.column_stack(test_data[fn] for fn in features) #np.column_stack([test_data['date_number']])

# model = SVR(C=1e3, gamma=0.1)
model = LinearRegression()
model.fit(X_train, y_train)

predict_result = model.predict(X_test)
actual_result = test_data[target_field]

# 计算准确率
total = 0
valid_count = 0
for i, (predict_result_item, actual_result_item) in enumerate(zip(predict_result, actual_result)):
    total += 1
    errorRate =  abs(actual_result_item - predict_result_item) / actual_result_item
    valid_count += 1 if (errorRate <= ErrorTolerance) else 0

accuracy = valid_count / total * 100

print("Features: {0}".format(features))
print('Error Tolerance: {0}'.format(ErrorTolerance))
print("Test count: {0}. Accuracy : {1:.2f}%".format(total, accuracy))
