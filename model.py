import joblib
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`should_run_async`", category=DeprecationWarning)


#LOAD DB
# Assuming the datasets and the .py file are in the same folder
current_folder = os.path.dirname(os.path.abspath(__file__))
# Path to the CSV file
file_path = os.path.join(current_folder, 'train.csv')
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Path to the JSON file release_evidences.json
file_path_evidences = os.path.join(current_folder, 'release_evidences.json')
# Load the release_evidences.json file
with open(file_path_evidences, "r") as f:
    evidence_data = json.load(f)
# Path to the JSON file release_conditions.json
file_path_conditions = os.path.join(current_folder, 'release_conditions.json')
# Load the release_conditions.json file
with open(file_path_conditions, "r") as f:
    condition_data = json.load(f)
# Drop rows with any missing values
df = df.dropna()

# Translate the Dataset to English
# Create a mapping from French to English condition names
french_to_english = {condition_data[key]["cond-name-fr"]: condition_data[key]["cond-name-eng"] for key in condition_data}
# Replace French condition names with English names in the dataset
df["PATHOLOGY"] = df["PATHOLOGY"].map(french_to_english)
# Extract English question list with evidence names
english_questions_with_evidence = {}

for evidence_name, evidence_info in evidence_data.items():
    # Check if evidence is binary
    if evidence_info["data_type"] == "B":
        for key, value in evidence_info.items():
            if key.startswith("question_en"):
                english_questions_with_evidence[value] = evidence_name

# Extract the 'cond-name-eng' values
pathologies = [condition["cond-name-eng"] for condition in condition_data.values()]

# Dictionary mapping pathologies to hospital departments
pathology_to_department = {
    "Spontaneous pneumothorax": "Pulmonology",
    "Cluster headache": "Neurology",
    "Boerhaave": "Gastroenterology",
    "Spontaneous rib fracture": "Orthopedics",
    "GERD": "Gastroenterology",
    "HIV (initial infection)": "Infectious Diseases",
    "Anemia": "Hematology",
    "Viral pharyngitis": "Otolaryngology",
    "Inguinal hernia": "General Surgery",
    "Myasthenia gravis": "Neurology",
    "Whooping cough": "Infectious Diseases",
    "Anaphylaxis": "Emergency Medicine",
    "Epiglottitis": "Otolaryngology",
    "Guillain-Barré syndrome": "Neurology",
    "Acute laryngitis": "Otolaryngology",
    "Croup": "Pediatrics",
    "PSVT": "Cardiology",
    "Atrial fibrillation": "Cardiology",
    "Bronchiectasis": "Pulmonology",
    "Allergic sinusitis": "Otolaryngology",
    "Chagas": "Infectious Diseases",
    "Scombroid food poisoning": "Emergency Medicine",
    "Myocarditis": "Cardiology",
    "Laryngospasm": "Otolaryngology",
    "Acute dystonic reactions": "Neurology",
    "Localized edema": "General Medicine",
    "SLE": "Rheumatology",
    "Tuberculosis": "Infectious Diseases",
    "Unstable angina": "Cardiology",
    "Stable angina": "Cardiology",
    "Ebola": "Infectious Diseases",
    "Acute otitis media": "Otolaryngology",
    "Panic attack": "Psychiatry",
    "Bronchospasm / acute asthma exacerbation": "Pulmonology",
    "Bronchitis": "Pulmonology",
    "Acute COPD exacerbation / infection": "Pulmonology",
    "Pulmonary embolism": "Pulmonology",
    "URTI": "Otolaryngology",
    "Influenza": "Infectious Diseases",
    "Pneumonia": "Pulmonology",
    "Acute rhinosinusitis": "Otolaryngology",
    "Chronic rhinosinusitis": "Otolaryngology",
    "Bronchiolitis": "Pediatrics",
    "Pulmonary neoplasm": "Oncology",
    "Possible NSTEMI / STEMI": "Cardiology",
    "Sarcoidosis": "Pulmonology",
    "Pancreatic neoplasm": "Oncology",
    "Acute pulmonary edema": "Cardiology",
    "Pericarditis": "Cardiology"
}


# FURTHER DATA PREPOCESSING
# Extract unique categorical, multi-choice, and binary evidence names
categorical_evidences = set()
multi_choice_evidences = set()
binary_evidences = set()

for evidence_name, evidence_info in evidence_data.items():
    if evidence_info["data_type"] == "C":
        categorical_evidences.add(evidence_name)
    elif evidence_info["data_type"] == "M":
        multi_choice_evidences.add(evidence_name)
    elif evidence_info["data_type"] == "B":
        binary_evidences.add(evidence_name)

# Drop unnecessary columns
df.drop(['DIFFERENTIAL_DIAGNOSIS', 'INITIAL_EVIDENCE'], axis=1, inplace=True)

# Perform one-hot encoding for 'SEX' column with a single binary column
df['SEX'] = (df['SEX'] == 'F').astype(int)

# Temporarlily dataset reduction
#df = df.head(200000)

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

df['EVIDENCES'] = df['EVIDENCES'].apply(eval)
df['EVIDENCES'] = split_and_keep_name(df['EVIDENCES'])
# Seperate the Evidences as new binary columns
mlb = MultiLabelBinarizer()
multi_choice_features = pd.DataFrame(mlb.fit_transform(df['EVIDENCES']),
                                     columns=mlb.classes_, index=df.index)
# Concatenate new binary columns with original dataframe
df = pd.concat([df, multi_choice_features], axis=1)
df.drop('EVIDENCES', axis=1, inplace=True)
# Columns to drop
columns_to_drop = sorted(categorical_evidences) + sorted(multi_choice_evidences)
# Drop columns from DataFrame
df = df.drop(columns=columns_to_drop)
# Define custom age ranges and corresponding labels
age_bins = [0, 13, 19, 31, 46, 61, float('inf')]
age_labels = ['0-12', '13-18', '19-30', '31-45', '46-60', '60<']
# Function to categorize AGE into predefined age_labels
def categorize_age(age):
    return pd.cut(age, bins=age_bins, labels=age_labels, right=False)
# Apply age categorization
df['AGE_CATEGORY'] = categorize_age(df['AGE'])
# Convert AGE_CATEGORY into binary columns
age_dummies = pd.get_dummies(df['AGE_CATEGORY'])
# Concatenate age_dummies with original dataframe
df = pd.concat([df, age_dummies], axis=1)
# Drop the original AGE and AGE_CATEGORY columns
df.drop(['AGE', 'AGE_CATEGORY'], axis=1, inplace=True)

df.head()


# PCA
# Extracting features (evidence columns)
features = df.drop(columns=['PATHOLOGY'])

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying PCA
pca = PCA()
pca.fit(scaled_features)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Create a DataFrame to visualize explained variance ratio by feature
variance_df = pd.DataFrame({'Feature': features.columns, 'Explained Variance Ratio': explained_variance_ratio})

# Sort features by explained variance ratio
variance_df = variance_df.sort_values(by='Explained Variance Ratio', ascending=False)

# Set a threshold for explained variance ratio
threshold = 0.001  # decided by plot above 'Explained Variance Ratio by Number of Components'

# Select features with explained variance ratio above the threshold
selected_features = variance_df[variance_df['Explained Variance Ratio'] > threshold]['Feature']

# Filter the original DataFrame to keep only selected features
selected_df = df[['PATHOLOGY'] + list(selected_features)]



#Association Rule Mining
# Preprocess the dataset
df_subset = df.drop(['SEX','0-12', '13-18', '19-30', '31-45', '46-60', '60<'], axis=1)
df_subset = pd.get_dummies(df_subset, columns=['PATHOLOGY'])

# Adjust min_support parameter
min_support = 0.02

# Apply FP-Growth algorithm with adjusted min_support
frequent_itemsets = fpgrowth(df_subset, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


def rules_mod(rules_df, pathology_categories, lift_threshold, confidence_threshold):
    """
    Filter association rules based on lift and confidence thresholds and relevant pathology categories.
    """
    filtered_rules = rules_df[
        (rules_df['lift'] >= lift_threshold) &
        (rules_df['confidence'] >= confidence_threshold) &
        (rules_df['consequents'].apply(lambda x: any(item.replace('PATHOLOGY_', '') in pathology_categories for item in x)))
    ]
    return filtered_rules

filtered_rules = rules_mod(rules, pathologies, lift_threshold=1.0, confidence_threshold=0.5)


#Label Encoding
# Separate features and target variable
X = df.drop('PATHOLOGY', axis=1)
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(df['PATHOLOGY'])

#Dataset Splitting
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#MODEL DEVELOPMENT

CV = StratifiedKFold(n_splits=2, random_state=0, shuffle=True)

# Instantiate the Logistic Regression model with increased max_iter
logistic_regression = LogisticRegression(random_state=42, max_iter=100000)

# Train the model
logistic_regression.fit(X_train, y_train)

# Make predictions
logistic_regression_pred = logistic_regression.predict(X_test)

# Calculate accuracy
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_pred)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)

# Calculate precision with 'macro' average
logistic_regression_precision = precision_score(y_test, logistic_regression_pred, average='macro')
print("Logistic Regression Precision:", logistic_regression_precision)

# Calculate recall with 'macro' average
logistic_regression_recall = recall_score(y_test, logistic_regression_pred, average='macro')
print("Logistic Regression Recall:", logistic_regression_recall)

# Calculate F1 score with 'macro' average
logistic_regression_f1_score = f1_score(y_test, logistic_regression_pred, average='macro')
print("Logistic Regression F1 Score:", logistic_regression_f1_score)


# Pre-diagnosis Application
# Function to get the related department
def get_department(pathology):
    return pathology_to_department.get(pathology, "Unknown Department")

def get_question_text(feature_index, questions_with_evidence, feature_names):
    # Create a mapping from feature name to index
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

    # Find the question text and evidence name corresponding to the feature index
    for question_text, evidence_name in questions_with_evidence.items():
        if feature_to_index[evidence_name] == feature_index:
            return question_text, evidence_name
    return None, None

def ask_questions_lg(log_reg, rules, questions_with_evidence, feature_names):
    def traverse_tree(sample):
        # Convert the sample to a DataFrame with the correct feature names
        sample_df = pd.DataFrame([sample], columns=feature_names)
        probas = log_reg.predict_proba(sample_df)[0]
        prediction = np.argmax(probas)
        return prediction, probas

    # Convert feature_names to a list
    feature_names = list(feature_names)

    # Create a mapping from feature name to index
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

    sample = np.zeros(len(feature_names))  # Initialize sample with zeros
    asked_questions = set()

    # Ask about sex
    sex = int(input("What is your sex? (female: 1, male: 0): "))
    sex_index = feature_to_index['SEX']
    sample[sex_index] = sex
    asked_questions.add(sex_index)

    # Ask about age
    age = int(input("How old are you? "))
    age_ranges = ['0-12', '13-18', '19-30', '31-45', '46-60', '60<']
    for age_range in age_ranges:
        age_index = feature_to_index[age_range]
        if age_range == '0-12' and age <= 12:
            sample[age_index] = 1
        elif age_range == '13-18' and 13 <= age <= 18:
            sample[age_index] = 1
        elif age_range == '19-30' and 19 <= age <= 30:
            sample[age_index] = 1
        elif age_range == '31-45' and 31 <= age <= 45:
            sample[age_index] = 1
        elif age_range == '46-60' and 46 <= age <= 60:
            sample[age_index] = 1
        elif age_range == '60<' and age > 60:
            sample[age_index] = 1
        else:
            sample[age_index] = 0
        asked_questions.add(age_index)

    while True:
        # Predict the result based on current answers
        prediction, probas = traverse_tree(sample)

        # Check if we can stop asking questions
        if max(probas) > 0.8 or len(asked_questions) >= len(feature_names):
            break

        # Find the next question to ask based on the rules
        for _, row in rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']

            # Skip if any antecedent has already been asked
            if any(feature_to_index[feat] in asked_questions for feat in antecedents):
                continue

            # Ask the next antecedent question
            for feature in antecedents:
                feature_index = feature_to_index[feature]
                if feature_index not in asked_questions:
                    question_text, _ = get_question_text(feature_index, questions_with_evidence, feature_names)
                    answer = int(input(f"{question_text} (0 or 1): "))
                    sample[feature_index] = answer
                    asked_questions.add(feature_index)
                    break

            # If we've asked a question, break the loop to re-evaluate
            if feature_index in asked_questions:
                break

    result_label = label_encoder.inverse_transform([prediction])[0]
    department = get_department(result_label)
    print(f"Predicted Result: {result_label} ... You should see the {department} department.")

# Example usage
feature_names = X.columns
#ask_questions_lg(logistic_regression, rules, english_questions_with_evidence, feature_names)

# Save Logistic Regression model
joblib.dump(logistic_regression, 'logistic_regression.pkl')

# Save Association Rules
joblib.dump(rules, 'rules.pkl')

# Save English Questions with Evidence
joblib.dump(english_questions_with_evidence, 'english_questions_with_evidence.pkl')

# Save Feature Names
joblib.dump(feature_names.tolist(), 'feature_names.pkl')

# Save Pathologies
joblib.dump(pathologies, 'pathologies.pkl')

# Save Feature Index Mapping (in case it's needed)
joblib.dump({name: idx for idx, name in enumerate(feature_names)}, 'feature_index.pkl')

# Save Pathology to Department Mapping
joblib.dump(pathology_to_department, 'pathology_to_department.pkl')
