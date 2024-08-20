import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

fraud_train = pd.read_csv('fraudTrain.csv')
fraud_test = pd.read_csv('fraudTest.csv')

print(fraud_train.info())
print(fraud_test.info())

fraud_train.fillna(method='ffill', inplace=True)
fraud_test.fillna(method='ffill', inplace=True)
label_encoder = LabelEncoder()

categorical_columns = ['merchant', 'category', 'first', 'last', 'gender', 'job']

for column in categorical_columns:
    fraud_train[column] = label_encoder.fit_transform(fraud_train[column])
    fraud_test[column] = label_encoder.transform(fraud_test[column])


fraud_train['trans_date_trans_time'] = pd.to_datetime(fraud_train['trans_date_trans_time'])
fraud_test['trans_date_trans_time'] = pd.to_datetime(fraud_test['trans_date_trans_time'])

fraud_train['day_of_week'] = fraud_train['trans_date_trans_time'].dt.dayofweek
fraud_train['hour_of_day'] = fraud_train['trans_date_trans_time'].dt.hour

fraud_test['day_of_week'] = fraud_test['trans_date_trans_time'].dt.dayofweek
fraud_test['hour_of_day'] = fraud_test['trans_date_trans_time'].dt.hour
fraud_train.drop(['trans_date_trans_time', 'cc_num', 'dob', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long'], axis=1, inplace=True)
fraud_test.drop(['trans_date_trans_time', 'cc_num', 'dob', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long'], axis=1, inplace=True)

scaler = StandardScaler()
numerical_columns = ['amt']

fraud_train[numerical_columns] = scaler.fit_transform(fraud_train[numerical_columns])
fraud_test[numerical_columns] = scaler.transform(fraud_test[numerical_columns])

X = fraud_train.drop('is_fraud', axis=1)
y = fraud_train['is_fraud']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)


y_pred_log_reg = log_reg.predict(X_val)


print("Logistic Regression:")
print(classification_report(y_val, y_pred_log_reg))
print("Accuracy:", accuracy_score(y_val, y_pred_log_reg))
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_val)


print("Decision Tree Classifier:")
print(classification_report(y_val, y_pred_tree))
print("Accuracy:", accuracy_score(y_val, y_pred_tree))

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

y_pred_forest = random_forest.predict(X_val)


print("Random Forest Classifier:")
print(classification_report(y_val, y_pred_forest))
print("Accuracy:", accuracy_score(y_val, y_pred_forest))

fpr, tpr, thresholds = roc_curve(y_val, y_pred_forest)
roc_auc = roc_auc_score(y_val, y_pred_forest)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

y_test_pred = random_forest.predict(fraud_test.drop('is_fraud', axis=1))

if 'is_fraud' in fraud_test.columns:
    print("Test Set Evaluation:")
    print(classification_report(fraud_test['is_fraud'], y_test_pred))
    print("Accuracy:", accuracy_score(fraud_test['is_fraud'], y_test_pred))

joblib.dump(random_forest, 'fraud_detection_model.pkl')
