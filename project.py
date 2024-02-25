import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# load dataset
df = pd.read_csv("kidney_disease.csv")
# drop the id column
df = df.drop(columns='id')

# Instantiate the encoder
encoder = LabelEncoder()

# convert 'pcv', 'wc', and 'rc' to numeric, coerce errors to NaN
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

# fill null values in numeric columns ('pcv', 'wc', 'rc') with their mean
df['pcv'] = df['pcv'].fillna(df['pcv'].mean())
df['wc'] = df['wc'].fillna(df['wc'].mean())
df['rc'] = df['rc'].fillna(df['rc'].mean())

# additional numeric columns that need null values filled
additional_numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
for column in additional_numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# fill null values in categorical columns with most frequent value (mode)
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# verify all null values are filled
null_values_final_check = df.isnull().sum()

# print final check for null values to confirm
print(null_values_final_check)

# Perform winsonization to remove outliers at the 5th and 95th percentiles
# create a list of numeric columns to winsonize
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
# create a for loop to winsonize each numeric column
for column in numeric_columns:
    q1 = df[column].quantile(0.05)
    q3 = df[column].quantile(0.95)
    df[column] = df[column].mask(df[column] < q1, q1)
    df[column] = df[column].mask(df[column] > q3, q3)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns

# Apply the encoder to each non-numeric column
for column in non_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

# perform wrapper method feature selection using RFE
# separate the target variable from the features
features = df.drop(columns='classification')
target = df['classification']


# perform RFE
model = LogisticRegression(solver='saga', max_iter=7000)
rfe = RFE(model)
fit = rfe.fit(features, target)

# display the features selected by RFE
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))

# Create a new dataframe with only selected features
selected_features = features.columns[fit.support_]
features_selected = features[selected_features]

# add back in the target
features_selected.loc[:, 'classification'] = target


# use pandas describe function to display summary statistics
print(features_selected.describe())

# save fully cleaned dataset to new CSV file
features_selected.to_csv(r"output.csv", index=False)

# Define Winsorization function
def winsorize_column(data, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_limit = data[column].quantile(lower_percentile)
    upper_limit = data[column].quantile(upper_percentile)
    data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
    return data

# Instantiate the encoder
encoder = LabelEncoder()

# Apply label encoding to non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

# Define Winsorization function
def winsorize_column(data, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_limit = data[column].quantile(lower_percentile)
    upper_limit = data[column].quantile(upper_percentile)
    data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
    return data
# Instantiate the encoder
encoder = LabelEncoder()

# Apply label encoding to non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

# Perform Winsorization
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for column in numeric_columns:
    df = winsorize_column(df, column)
# perform RFE
features = df.drop(columns='classification')
target = df['classification']
model = LogisticRegression(solver='saga', max_iter=7000)
rfe = RFE(model)
fit = rfe.fit(features, target)
# Display summary statistics for the dataset after Winsorization
selected_features = features.columns[fit.support_]
features_selected = features[selected_features]
features_selected['classification'] = target
print(features_selected.describe())


# Adjust the x and y-axis limits in boxplots
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot of Numeric Columns Before Winsorization')
plt.xticks(rotation=45)
plt.ylim(0, 160)  # Adjust the y-axis limit
plt.yticks(range(0, 161, 15))  # Set y-axis tick intervals to 15
plt.show()

# Visualize outliers after Winsorization with adjusted axis limits
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot of Numeric Columns After Winsorization')
plt.xticks(rotation=45)
plt.ylim(0, 160)  # Adjust the y-axis limit
plt.yticks(range(0, 161, 15))  # Set y-axis tick intervals to 15
plt.show()

# Display summary statistics for the dataset after Winsorization
print(features_selected.describe())
