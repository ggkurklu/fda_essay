import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('drive/MyDrive/dataset.csv')
df.head()

# TASK 1-1
numeric_df = df.select_dtypes(include=['number'])

for column in numeric_df.columns:
    non_zero_data = numeric_df[column][numeric_df[column] != 0]

    mean = np.mean(non_zero_data)
    median = np.median(non_zero_data)
    std_dev = np.std(non_zero_data)

    # Print the results for each column
    print(f"Column: {column}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}\n")

# TASK 1-2
categorical_cols = df.select_dtypes(include=['object'])
for column in categorical_cols.columns:
    df[column] = df[column].str.lower()
    frequency_counts = categorical_cols[column].value_counts()

    print(f"Frequencies for column: {column}")
    print(frequency_counts)
    print("\n")

#TASK 1-4 , 1-5
def clean_dataset_(data):
    """
    Cleans the given DataFrame by handling missing values,
    removing duplicates, and filtering out outliers.
    """
    # print("Data before cleaning:")
    # print(data.shape)
    # print("Data types:\n", data.dtypes)

    # Handling missing values in the dataset
    missing_values = data.isnull().sum()
    # print("Missing values in each column before handling:")
    # print(missing_values)

    # Impute missing values for numeric columns with mean
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Impute missing values for categorical columns with mode
    categorical_cols = data.select_dtypes(include='object').columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Removing any duplicates
    data.drop_duplicates(inplace=True)

    # Detect outliers using z-scores
    mean_age = data['age'].mean()
    std_age = data['age'].std()
    z_scores_age = (data['age'] - mean_age) / std_age

    # Cap outliers at 3 standard deviations
    threshold = 3
    data['age_capped'] = np.where(z_scores_age > threshold, mean_age + threshold * std_age,
                                   np.where(z_scores_age < -threshold, mean_age - threshold * std_age, data['age']))

    # Display the result for age
    print("max age before cap" , data['age'].max())
    print("min age before cap", data['age'].min())

    print("max age after cap" , data['age_capped'].max())
    print("min age after cap" , data['age_capped'].min())


    # Treat rare job categories as 'Other'
    job_counts = data['job'].value_counts()
    print("initial jobs")
    print(job_counts)

    # Set a threshold for rare jobs
    threshold = 1000  # remove jobs under 1000 to 'Other' due to insignificance in the study.
    data['job_grouped'] = data['job'].apply(lambda x: x if job_counts[x] > threshold else 'Other')

    # Display the result for job
    job_grouped_counts = data['job_grouped'].value_counts()
    print("Jobs after Grouping =< 1000")
    print(job_grouped_counts)

    # Cap duration at the 99th percentile
    percentile_99_duration = data['dur'].quantile(0.99)
    data['duration_capped'] = np.where(data['dur'] > percentile_99_duration, percentile_99_duration, data['dur'])

    # Display the result for duration
    print(data[['dur', 'duration_capped']].head())

    # Check for missing values after handling
    missing_values_after = data.isnull().sum()
    # print("Missing values in each column after handling:")
    # print(missing_values_after)

    # print("Data types after cleaning:")
    # print(data.dtypes)

    # Return cleaned dataset
    return data


# Use the cleaned dataset
cleaned_df = clean_dataset_(df)

yes_counts = cleaned_df[cleaned_df['y'] == 'yes']

# Step 2: Group by num_calls and count occurrences

# Step 3: Identify the num_calls value with the maximum count of 'yes'
print(yes_counts)

# TASK 1-7
# Convert target variable to a categorical type
cleaned_df.loc[:, 'y'] = cleaned_df['y'].astype('category')
print(cleaned_df['age_capped'].max())

filtered_df = cleaned_df[(cleaned_df['age_capped'] >= 70) & (cleaned_df['age_capped'] <= 72.79447766217368) & (cleaned_df['y'] == 'yes')]
# Get the count of these rows
count_yes = filtered_df.shape[0]
print(f"Number of 'yes' between age 65 and 72: {count_yes}")

# Filter data for 'yes' conversions
yes_data = cleaned_df[cleaned_df['y'] == 'yes']

print(yes_data.shape)
print(yes_data.columns)

# Set up the plotting area with subplots
fig, axes = plt.subplots(5, 2, figsize=(20, 15), sharex=False, sharey=False)
fig.suptitle('Feature Distribution for Conversions (y == yes)', fontsize=16)

# List of features to plot
features = ['age_capped', 'job_grouped', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'duration_capped', 'num_calls', 'prev_outcome']
plot_titles = ['Age', 'Job', 'Marital Status', 'Education Qualification', 'Call Type', 'Day of Month', 'Month', 'Duration', 'Number of Calls', 'Previous Outcome']

for ax, feature, title in zip(axes.flatten(), features, plot_titles):
    if yes_data[feature].dtype == 'object':
        sns.countplot(data=yes_data, x=feature, ax=ax)
    else:
        sns.histplot(yes_data[feature], ax=ax, bins=30, kde=True, color='skyblue')

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # layout adjusment
plt.show()

#decision tree
label_encoder = LabelEncoder()
cleaned_df['y'] = label_encoder.fit_transform(cleaned_df['y'])

# Select features (X) and target (y)
X = cleaned_df.drop(columns=['y', 'age_capped', 'duration_capped'])
y = cleaned_df['y']  # This will now be [0, 1]

# Print the transformed values for verification
print("Unique values in y after encoding:", y.unique())

# Diagnostic checks
print(f"Data type of y: {y.dtype}")
print(f"Unique values in y: {y.unique()}")

# Convert categorical variables to numeric (if not done yet)
X = pd.get_dummies(X, drop_first=True)

# Check for any NaN values
print(f"NaN values in X: {X.isnull().sum().sum()}")
print(f"NaN values in y: {y.isnull().sum()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of the split datasets
print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optionally reduce dimensionality with PCA
pca = PCA(n_components=0.95)  # Adjust this to retain 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=2, random_state=42)

# Fit the model
try:
    dt_classifier.fit(X_train_pca, y_train)
    print("Decision Tree model fitted successfully.")
    print(f"Depth of the tree: {dt_classifier.get_depth()}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Predict on the test set
y_pred = dt_classifier.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy of Decision Tree: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
class_names = [str(name) for name in label_encoder.classes_]
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=class_names)
plt.title("Decision Tree Visualization")
plt.show()

# Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, penalty='l2')

# Fit the model
try:
    logistic_model.fit(X_train_pca, y_train)
    print("Logistic Regression model fitted successfully.")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Predict on the test set
y_pred = logistic_model.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy of Logistic Regression: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy of Random Forest: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize Feature Importances
plt.figure(figsize=(12, 6))
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
