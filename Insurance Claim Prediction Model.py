#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# Load the dataset
df = pd.read_csv('/Users/shubham/Documents/Insurance/car_insurance_claim.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


num_unique_values = df.nunique()

# Print the number of unique values for each column
print("Number of unique values for each column:")
print(num_unique_values)


# In[7]:


# Create a SimpleImputer object with strategy='mean'
imputer_mean = SimpleImputer(strategy='mean')

# Identify columns with missing values
columns_with_missing = df.columns[df.isnull().any()]

# Replace missing values with mean for numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
columns_to_impute = list(set(columns_with_missing) & set(numerical_columns))
df[columns_to_impute] = imputer_mean.fit_transform(df[columns_to_impute])

# Check if there are any missing values left
missing_values_after_imputation_mean = df.isnull().sum()
print("Missing values after imputation with mean:")
print(missing_values_after_imputation_mean)


# In[8]:


# Find columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()]

# Display the columns with missing values and their counts
print("Columns with missing values:")
print(df[columns_with_missing_values].isnull().sum())


# In[ ]:





# In[9]:


# Data preprocessing
columns_to_convert = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
for col in columns_to_convert:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    if 'z_' in df[col].unique():
        df[col] = df[col].replace('z_', '', regex=True)
        df[col] = df[col].astype('category')


# In[10]:


# Visualize distributions and correlation matrix
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[12]:


# Feature engineering
winsorized_columns = ['INCOME', 'HOME_VAL', 'OLDCLAIM', 'CLM_AMT']
for col in winsorized_columns:
    percentile_95 = np.percentile(df[col].dropna(), 95)
    df[col] = np.where(df[col] > percentile_95, percentile_95, df[col])

clipped_columns = ['AGE', 'YOJ', 'TRAVTIME', 'CAR_AGE']
for col in clipped_columns:
    threshold = df[col].quantile(0.95)
    df[col] = df[col].clip(upper=threshold)


# In[ ]:





# In[13]:


df.head(5)


# In[14]:


df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 25, 40, 60, np.inf], labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
df['INCOME_PER_YEAR'] = df['INCOME'] / df['YOJ']
df['HOME_VAL_INCOME_RATIO'] = df['HOME_VAL'] / df['INCOME']
df['OLDCLAIM_INCOME_RATIO'] = df['OLDCLAIM'] / df['INCOME']
df['CLAIM_FREQ_AMT_RATIO'] = df['CLM_FREQ'] / df['CLM_AMT']
df['MARRIED_WITH_KIDS'] = np.where((df['MSTATUS'] == 'Married') & (df['PARENT1'] == 'Yes'), 1, 0)


# In[15]:


old_variables = ['AGE', 'INCOME', 'YOJ', 'HOME_VAL', 'OLDCLAIM', 'CLM_FREQ', 'MSTATUS', 'PARENT1']
df.drop(old_variables, axis=1, inplace=True)


# In[16]:


string_columns = ['GENDER', 'EDUCATION', 'OCCUPATION', 'URBANICITY', 'CAR_TYPE']
for col in string_columns:
    df[col] = df[col].str.replace('z_', '')


# In[17]:


df.head(5)


# In[18]:


df['CLAIM_FREQ_AMT_RATIO'].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
mean_ratio = df['CLAIM_FREQ_AMT_RATIO'].mean()
df['CLAIM_FREQ_AMT_RATIO'].fillna(mean_ratio, inplace=True)

df['BIRTH_YEAR'] = '19' + df['BIRTH'].str[-2:]
df['BIRTH_YEAR'] = df['BIRTH_YEAR'].astype(int)
current_year = 2024
df['AGE'] = current_year - df['BIRTH_YEAR']
df.drop(columns=['BIRTH', 'BIRTH_YEAR'], inplace=True)


# In[ ]:





# In[19]:





# In[20]:


# Encoding categorical variables
df['CAR_USE'] = df['CAR_USE'].replace({'Commercial': 1, 'Private': 0})
df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})
df['RED_CAR'] = df['RED_CAR'].replace({'yes': 1, 'no': 0})
df['REVOKED'] = df['REVOKED'].replace({'Yes': 1, 'No': 0})

df['CAR_USE'] = df['CAR_USE'].astype(int)
df['RED_CAR'] = df['RED_CAR'].astype(int)
df['REVOKED'] = df['REVOKED'].astype(int)
df['REVOKED'] = df['GENDER'].astype(int)


# In[21]:


df.head(5)


# In[22]:


df.info()


# In[23]:


# Perform one-hot encoding on categorical variables
df_encoded = pd.get_dummies(df, columns=['GENDER', 'EDUCATION', 'OCCUPATION', 'URBANICITY', 'AGE_GROUP'])


# In[24]:


df_encoded.head()


# In[25]:


df_encoded.drop(columns=['ID', 'CAR_TYPE'], inplace=True)
df_encoded.head()


# In[26]:


# Find rows with infinite values
rows_with_infinite = df_encoded[~np.isfinite(df_encoded).all(axis=1)]

# Remove rows with infinite values
df_cleaned = df_encoded[np.isfinite(df_encoded).all(axis=1)]

df_cleaned.info(5)


# In[ ]:





# In[28]:


df_cleaned.head(5)


# In[ ]:





# In[ ]:





# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[30]:


# Split the data into features (X) and target variable (y)
X = df_cleaned.drop(columns=['CLAIM_FLAG'])
y = df_cleaned['CLAIM_FLAG']


# In[31]:


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Initialize models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}


# In[33]:


# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}


# In[35]:


# Initialize the logistic regression model with a higher max_iter value
logistic_regression_model = LogisticRegression(max_iter=1000)

# Train and evaluate the model
logistic_regression_model.fit(X_train, y_train)
y_pred = logistic_regression_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Store the results
results['Logistic Regression'] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}


# In[36]:


print(results)


# In[34]:


# Summarize results
results_df = pd.DataFrame(results)
print(results_df)


# In[38]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[39]:


# Loop through each model
for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()

    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

    # Print results
    print(f"Model: {name}")
    print(f"Cross-Validation Accuracy: {cv_accuracy}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50)

    # Perform hyperparameter tuning using GridSearchCV
    if name in ['Random Forest', 'Logistic Regression', 'Support Vector Machine']:
        params = {}  # Define the hyperparameters to tune
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_score}")
        print("="*50)
        


# In[40]:


from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the classifier to your training data
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Pair feature names with their importance scores
feature_importance_dict = dict(zip(X_train.columns, feature_importances))

# Sort features by importance score (descending order)
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the top n features
n_top_features = 5  # Change this value as needed
print(f"Top {n_top_features} features:")
for feature, importance in sorted_features[:n_top_features]:
    print(f"{feature}: {importance}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




