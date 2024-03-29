import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Train 
data_train = pd.read_csv("test.csv")
# data_train.info()
# print(data_train.head(3))


# Test 
# data_test = pd.read_csv("test.csv")
# print(data_test.head(3))
# data_test.info()

# Validate 
# data_validate = pd.read_csv("validate.csv")
# print(data_validate.head(3))
# data_validate.info()

# Perform one-hot encoding for 'SEX' column
data_train = pd.get_dummies(data_train, columns=['SEX'], dtype= int)

# Eliminate value embedded with '_@_'
def split_and_keep_name(array):
    new_array = []
    for sublist in array:
        new_sublist = []
        for item in sublist:
            if '_@_' in item:
                new_sublist.append(item.split('_@_')[0])
            else:
                new_sublist.append(item)
        new_array.append(new_sublist)
    return new_array

data_train['EVIDENCES'] = data_train['EVIDENCES'].apply(eval)
data_train['EVIDENCES'] = split_and_keep_name(data_train['EVIDENCES'])

# Seperate the Evidences as new bianry columns
mlb = MultiLabelBinarizer()
multi_choice_features = pd.DataFrame(mlb.fit_transform(data_train['EVIDENCES']), columns=mlb.classes_, index=data_train.index)

data_train = pd.concat([data_train, multi_choice_features], axis=1)
data_train.drop('EVIDENCES', axis=1, inplace=True)

# Drop the 'DIFFERENTIAL_DIAGNOSIS' and 'INITIAL_EVIDENCE' column
data_train.drop('DIFFERENTIAL_DIAGNOSIS', axis=1, inplace=True)
data_train.drop('INITIAL_EVIDENCE', axis=1, inplace=True)

data_train.info()
print(data_train.head(3))

###############
# MODEL 

# Step 1: Separate features and target variable
X = data_train.drop('PATHOLOGY', axis=1)  # Features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data_train['PATHOLOGY'])
# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train decision tree and random forest classifiers
dt_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

# Step 4: Evaluate models
dt_pred = dt_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
