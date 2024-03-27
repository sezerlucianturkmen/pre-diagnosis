import pandas as pd

# Train 
data_train = pd.read_csv("train.csv")
data_train.info()
print(data_train.head(3))


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

print(data_train.head(3))

