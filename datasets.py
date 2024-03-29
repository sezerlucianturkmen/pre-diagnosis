import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_train = pd.read_csv("train.csv")
data_train = data_train.sample(n=10000, random_state=42)
data_train.info()

# Perform one-hot encoding for 'SEX' column
data_train = pd.get_dummies(data_train, columns=['SEX'], dtype=int)

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

# Seperate the Evidences as new binary columns
mlb = MultiLabelBinarizer()
multi_choice_features = pd.DataFrame(mlb.fit_transform(data_train['EVIDENCES']),
                                     columns=mlb.classes_, index=data_train.index)

# Concatenate new binary columns with original dataframe
data_train = pd.concat([data_train, multi_choice_features], axis=1)
data_train.drop('EVIDENCES', axis=1, inplace=True)

# Drop unnecessary columns
data_train.drop(['DIFFERENTIAL_DIAGNOSIS', 'INITIAL_EVIDENCE'], axis=1, inplace=True)

# See current dataframe
print(data_train.head(5))

# Separate features and target variable
X = data_train.drop('PATHOLOGY', axis=1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data_train['PATHOLOGY'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree and random forest classifiers
dt_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get best estimator
rf_classifier = grid_search.best_estimator_

# Train decision tree classifier
dt_classifier.fit(X_train, y_train)

# Evaluate models
dt_pred = dt_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Confusion matrix for Random Forest classifier
rf_cm = confusion_matrix(y_test, rf_pred)

# Plot confusion matrix for Random Forest classifier
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature importance for Random Forest classifier
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
