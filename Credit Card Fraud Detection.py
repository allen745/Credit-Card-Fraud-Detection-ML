# data --> data preprocessing --> data analysis --> train test split --> logistic regression--> evalution

# importing dependencies

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#  loading data to pandas dataframe

credit_card_data = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\creditcard.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(credit_card_data.head())

# credit cars info

print(credit_card_data.info())
print(credit_card_data.describe())

# checking null values
print(credit_card_data.isnull().sum())

# distribution of legit transaction & fraud transection
print(credit_card_data['Class'].value_counts())

# the data set is highly unbalanced
# 0--> normal transection 1 --> fraud transection

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#  statical measures of data
legit.Amount.describe()
fraud.Amount.describe()

# compare the value for both transection

credit_card_data.groupby('Class').mean()

# building a sample dataset containing similar distribution of modal transection and fraudulent transection
# number of fraudulent transection --> 492

legit_sample = legit.sample(n=492)

# concatenating two dataframe

new_dataset = pd.concat([legit_sample, credit_card_data], axis=0)

print(new_dataset.head())

new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

# splitting in to features and target

x = new_dataset.drop('Class', axis=1)
y = new_dataset['Class']

# splitting data in to test data and train data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# modal training

model = LogisticRegression()

# training the logistic regression modal with train data
model.fit(x_train, y_train)
# modal evalution
# accuracy score
# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy on training data: ", training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy on testing data: ", test_data_accuracy)

