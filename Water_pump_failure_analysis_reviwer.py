#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss, ClusterCentroids
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve
import shap
from tqdm import tqdm

# Global parameters 
RANDOMSTATE = 101


# ### Importing the dataset

# In[ ]:


# Importing the dataset
data = pd.read_csv('sensor.csv')  # Make sure the file is in the same working directory
sensor_data = data.copy()
sensor_data.head()


# # Exploratory Data Analysis

# ### Display dataset information

# In[ ]:


# Check the 'machine_status' category
sensor_data['machine_status'].unique()


# ### Descriptive statistics¶

# In[ ]:


# Perform descriptive statistics
pd.set_option('display.max_columns', None)
sensor_data.describe()


# ### Check amount of missing data

# In[ ]:


# Calculating the number of missing values per column
total_missing = sensor_data.isnull().sum()

# Calculating the percentage of missing values per column
percent_missing = (total_missing / sensor_data.shape[0]) * 100

# Displaying the percentage of missing data per column
print("Percentage of missing data per column:")
print(percent_missing)

# Displaying columns with more than 20% missing data
print("Columns with more than 20% missing data:")
print(percent_missing[percent_missing > 20])


# ### Missing data handling

# The Unnamed: 0 column repeats the index of the dataset. The 'sensor_00' and 'sensor_51' columns are missing about 6% to 7% of their values. The 'sensor_50' column is missing 35% of its values. Finally, sensor_15 has no values.

# In[ ]:


sensor_data = sensor_data.drop(['Unnamed: 0', 'sensor_00', 'sensor_15', 'sensor_50', 'sensor_51', ], axis=1)


# ### Analyzing Class Distribution of Machine Status

# In[ ]:


# Counting Occurrences of Each Machine Status
category_counts = sensor_data['machine_status'].value_counts()
# Display category count
print(category_counts)


# ### Visualizing Class Distribution of Machine Status with Pie Chart

# In[ ]:


# Define custom colors using the 'pastel' palette
colors = [sns.color_palette("pastel")[0], sns.color_palette("pastel")[2], sns.color_palette("pastel")[3]]

# Count the number of occurrences for each category in the 'machine_status' column
category_counts = sensor_data['machine_status'].value_counts()

# Plotting the pie chart
plt.pie(category_counts, explode=(0.2, 0, 0), autopct='%1.3f%%', wedgeprops={'edgecolor': 'black'},
        counterclock=False, shadow=True, startangle=25, pctdistance=1.2, radius=1.3, colors=colors,
        textprops={'fontsize': 15})

# Adding legend using the index of category counts
plt.legend(category_counts.index, loc='lower right', fontsize=10)

# Set the title of the plot
plt.title('Percentage of Machine Status', fontsize=20)

# Adjust layout to avoid clipping elements
plt.tight_layout()

# Display the plot
plt.show()


# ### Visualize possible failure patterns

# Let's try to identify the trends and patterns of pump recovery and breakdown.

# In[ ]:


import matplotlib.pyplot as plt

# Set Seaborn style
sns.set(style="darkgrid")

# Set Times New Roman as the default font for plots
plt.rcParams['font.family'] = 'Times New Roman'

# Separate data instances based on the 'machine_status' column
sensor_cols = sensor_data.iloc[:, 1:49]
broken_rows = sensor_data[sensor_data['machine_status'] == 'BROKEN']
recovery_rows = sensor_data[sensor_data['machine_status'] == 'RECOVERING']
normal_rows = sensor_data[sensor_data['machine_status'] == 'NORMAL']
machine_status_col = sensor_data['machine_status']

# Plotting the sensor time series
for sensor in sensor_cols:
    plt.figure(figsize=(13, 2.5))  # Adjust height as needed
    
    plt.plot(normal_rows[sensor], label='NORMAL', linestyle='none', marker='o', markersize=1)
    plt.plot(recovery_rows[sensor], label='RECOVERING', linestyle='none', marker='o', color='yellow', markersize=3)
    plt.plot(broken_rows[sensor], label='BROKEN', linestyle='none', marker='X', color='red', markersize=10)
    
    plt.title(sensor, fontsize=15)
    plt.xlabel('Timestamp (min)', fontsize=13)
    plt.ylabel('Sensor Values', fontsize=13)
    plt.xlim((-1000, 177000))
    plt.legend(prop={'size': 10})
    
    # Adjust tick font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.savefig(f'plot_sensor_{sensor}.png', dpi=600, format='png', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


# First, we see that many of the graphs show large changes around the 'x' marker time 20,000-75,000, as well as around time 130,000. Some sensors drop to very low values, some increase, while others (which are usually consistent) vibrate quite a bit. We see that at these times the machine has failed and is recovering at least once. This could be an indicator to predict failure. This leads to the next task: working with missing data.

# # Data preprocessing

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.compose import make_column_selector as selector

# 1. Load and fill missing data
sensor_datap = sensor_data.fillna(method='ffill')

# 2. Merge 'RECOVERING' into 'NORMAL' in the target column
sensor_datap['machine_status'] = sensor_datap['machine_status'].replace('RECOVERING', 'NORMAL')

# 3. Separate features and target
X = sensor_datap.drop(['machine_status'], axis=1)
y = sensor_datap['machine_status']

# 4. Split dataset into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 6. Select numeric columns and apply normalization
numeric_selector = selector(dtype_include=np.number)
numeric_features = numeric_selector(X_train)

normalizer = Normalizer(norm='l2')
X_train_normalized = normalizer.fit_transform(X_train[numeric_features])
X_test_normalized = normalizer.transform(X_test[numeric_features])

# Convert normalized arrays back to pandas DataFrames
X_train_df = pd.DataFrame(X_train_normalized, columns=numeric_features, index=X_train.index)
X_test_df = pd.DataFrame(X_test_normalized, columns=numeric_features, index=X_test.index)


# ### Dimensionality reduction with PCA

# In[ ]:


from sklearn.decomposition import PCA

# Set Seaborn style
sns.set(style="darkgrid")

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.99)  # Keep components that explain 95% of the variance

# Fit and transform on training data
X_train_pca = pca.fit_transform(X_train_df)

# Transform test data using the same PCA model
X_test_pca = pca.transform(X_test_df)

# Optional: Convert back to pandas DataFrame for easier post-processing or analysis
X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train_df.index)
X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test_df.index)

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Number of components explaining 99% of the variance: {pca.n_components_}")
print(f"\nExplained variance ratio per principal component:\n{explained_variance}")

# Optional: Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
print(f"\nCumulative explained variance after each component:\n{cumulative_variance}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA - Cumulative Explained Variance')
plt.axhline(y=0.99, color='r', linestyle='--', label='99% variance threshold')
plt.grid(True)
plt.legend()
plt.show()


# ### Analysis of Variables in Each Principal Component (PC)

# In[ ]:


# Create DataFrame with the weights (loadings) of each variable in the PCs
loadings = pd.DataFrame(
    pca.components_.T,  # Transpose the matrix to have variables as rows
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_train_df.columns
)

# MosShow the most important variables for each PC
print("\nMost significant variables in each Principal Component:")
for pc in loadings.columns:
    # Get the 5 variables with the highest absolute weight in this PC
    top_vars = loadings[pc].abs().sort_values(ascending=False).head(5)
    print(f"\n{pc}:")
    print(top_vars)


# ### Graphical View of Contributions

# In[ ]:


import seaborn as sns

# Create bar chart for first 5 PCs
n_pcs_to_plot = min(5, pca.n_components_)
fig, axes = plt.subplots(n_pcs_to_plot, 1, figsize=(12, 4*n_pcs_to_plot))

for i, ax in enumerate(axes):
    pc_name = f'PC{i+1}'
    # Get the 10 most important variables for this PC
    top_vars = loadings[pc_name].abs().sort_values(ascending=False).head(10)
    
    # Plot
    loadings.loc[top_vars.index, pc_name].plot.bar(ax=ax, color='skyblue')
    ax.set_title(f'Top 10 Variables in {pc_name} (Var. Explained: {explained_variance[i]:.2%})')
    ax.set_ylabel('Weight on PC')
    ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()


# ### Correlation Matriz

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set Times New Roman as the default font
plt.rcParams['font.family'] = 'Times New Roman'

# Ensure PCA-transformed data is in DataFrame format
pca_columns = [f"PC{i+1}" for i in range(X_train_pca_df.shape[1])]
X_train_pca_df.columns = pca_columns  # Rename columns as PC1, PC2, etc.

# Calculate the correlation matrix
correlation_matrix = X_train_pca_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='Blues',
    square=True,
    linewidths=0.5,
    linecolor='gray',
    vmin=-1,
    vmax=1,
    cbar_kws={"shrink": 0.75},
    annot_kws={"size": 8}
)

# Add title and adjust layout
plt.title("Correlation Matrix of Principal Components", fontsize=14, pad=20)

# Adjust axis labels (optional: replace with most important sensors)
try:
    most_important_sensors = [sensor_datap[columns_to_normalize].columns[np.argmax(np.abs(pca.components_[i]))] 
                               for i in range(pca.n_components_)]
    
    ax.set_xticklabels(most_important_sensors, rotation=45, fontsize=10)
    ax.set_yticklabels(most_important_sensors, rotation=0, fontsize=10)
except Exception as e:
    print("Could not apply feature labels to axes:", e)

# Final layout adjustments and save figure
plt.tight_layout()
plt.savefig('correlation_matrix_pca.png', dpi=600, bbox_inches='tight')

# Display the plot
plt.show()


# # Model development

# In[ ]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

# Define the models
models = [
    RandomForestClassifier(random_state=42, n_jobs=-1),
    SVC(kernel='rbf', probability=True, random_state=42),
    KNeighborsClassifier(n_neighbors=5),
]

# Set up plot
fig, axes = plt.subplots(1, len(models), figsize=(28, 6))
fig.suptitle('Confusion Matrix - Test Set', fontsize=30)
labels = ['NORMAL', 'BROKEN']

results_df = pd.DataFrame()

for j, model in enumerate(models):
    print('Fitting', model)
    start_time = time.time()
    
    # Train model
    model.fit(X_train_pca_df, y_train_encoded)
    
    end_time = time.time()
    process_time = end_time - start_time

    # Cross-validation setup
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    
    scoring = {
        'f1': 'f1_macro',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'roc_auc': 'roc_auc'
    }
    
    cross_val_scores = cross_validate(model, X_train_pca_df, y_train_encoded, cv=cv, scoring=scoring, n_jobs=-1)

    # Mean scores from cross-validation
    f1_test_cv = round(cross_val_scores['test_f1'].mean(), 4)
    precision_test_cv = round(cross_val_scores['test_precision'].mean(), 4)
    recall_test_cv = round(cross_val_scores['test_recall'].mean(), 4)
    roc_auc_test_cv = round(cross_val_scores['test_roc_auc'].mean(), 4)

    # Predict on test set
    y_pred = model.predict(X_test_pca_df)

    # Compute metrics manually to double-check
    f1_manual = round(f1_score(y_test_encoded, y_pred, average='macro'), 4)
    precision_manual = round(precision_score(y_test_encoded, y_pred, average='macro'), 4)
    recall_manual = round(recall_score(y_test_encoded, y_pred, average='macro'), 4)
    roc_auc_manual = round(roc_auc_score(y_test_encoded, model.predict_proba(X_test_pca_df)[:, 1]), 4)

    # Store results
    score_df = pd.DataFrame({
        'f1': [f1_test_cv],
        'precision': [precision_test_cv],
        'recall': [recall_test_cv],
        'roc_auc': [roc_auc_test_cv],
        'Processing Time (s)': [process_time]},
        index=[str(model).split('(')[0]])

    results_df = pd.concat([results_df, score_df])

    print(f'Cross val ROC AUC: {roc_auc_test_cv}')
    print(f'Cross val F1 macro: {f1_test_cv}')
    print(f'Cross val Precision macro: {precision_test_cv}')
    print(f'Cross val Recall macro: {recall_test_cv}')
    print(f'Processing time: {round(process_time, 2)} seconds')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 30, "weight": "bold"}, ax=axes[j])

    axes[j].set_xticklabels(labels, fontsize=20)
    axes[j].set_yticklabels(['', ''], fontsize=20)
    axes[j].set_xlabel('Prediction \n ROC AUC = ' + str(roc_auc_manual), fontsize=20)

    if j == 0:
        axes[j].set_yticklabels(labels, fontsize=20)
        axes[j].set_ylabel('True Label', fontsize=25)

    axes[j].set_title(str(model).split('(')[0], fontsize=20)

plt.tight_layout()
plt.savefig('confusion_matrix_model_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

# Sort and display final results
results_df = results_df.sort_values(by='f1', ascending=False)
results_df.round(4)


# ### Evaluation of class balancing techniques

# In[ ]:


from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, TomekLinks, NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.pipeline import Pipeline as imbpipeline
import pandas as pd
import time

# Define models
models = [
    RandomForestClassifier(random_state=42),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='rbf', probability=True, random_state=42)
]

# Define sampling techniques
sampling_methods = [
    RandomOverSampler(random_state=42),
    BorderlineSMOTE(random_state=42),
    ClusterCentroids(random_state=42),
    TomekLinks(n_jobs=-1),
    NearMiss(version=3)
]

# Results DataFrame
results_df = pd.DataFrame(columns=[
    'Model', 'Sampling Technique', 'F1 Score', 'AUC-ROC', 'Precision', 'Recall', 'Processing Time'
])

# Evaluation loop
for model in models:
    for sampler in sampling_methods:
        start_time = time.time()

        # Create pipeline with resampling and classifier
        pipeline = imbpipeline([
            ('sampler', sampler),
            ('classifier', model)
        ])

        # Fit on PCA-transformed training data
        pipeline.fit(X_train_pca_df, y_train_encoded)

        # Predict on test set
        y_pred = pipeline.predict(X_test_pca_df)

        # Compute metrics
        f1 = round(f1_score(y_test_encoded, y_pred, average="macro"), 4)
        precision = round(precision_score(y_test_encoded, y_pred, average="macro"), 4)
        recall = round(recall_score(y_test_encoded, y_pred, average="macro"), 4)
        try:
            roc_auc = round(roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test_pca_df)[:, 1]), 4)
        except AttributeError:
            # Some models like SVM without probability don't have predict_proba
            roc_auc = np.nan

        processing_time = round(time.time() - start_time, 2)

        # Store results
        score_df = pd.DataFrame({
            'Model': [str(model).split('(')[0]],
            'Sampling Technique': [str(sampler).split('(')[0]],
            'F1 Score': [f1],
            'AUC-ROC': [roc_auc],
            'Precision': [precision],
            'Recall': [recall],
            'Processing Time': [processing_time]
        })

        results_df = pd.concat([results_df, score_df], ignore_index=True)

# Sort by F1 Score
results_df = results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
results_df.round(4)


# ### Hyperparameter optimization

# In[ ]:


from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the pipeline with resampling and classification
pipeline = imbpipeline([
    ('sampler', ClusterCentroids(sampling_strategy='auto', random_state=42)),
    ('rf_classifier', RandomForestClassifier(random_state=42))
])

# Define the hyperparameter search space
param_grid = {
    'rf_classifier__n_estimators': [100, 200],              # Number of trees in the forest
    'rf_classifier__max_depth': [None, 10, 20],             # Maximum depth of each tree
    'rf_classifier__min_samples_split': [2, 5, 10],         # Minimum number of samples required to split a node
    'rf_classifier__min_samples_leaf': [1, 2, 4],           # Minimum number of samples required at each leaf node
    'rf_classifier__max_features': ['sqrt', 'log2'],        # Number of features considered for splitting
    'rf_classifier__criterion': ['gini', 'entropy'],        # Quality measure for splits
    'rf_classifier__bootstrap': [True, False]               # Whether to use bootstrap when building trees
}

# Configure GridSearchCV with stratified cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,                   # Increased to 5 folds for more robust evaluation
    verbose=1,
    n_jobs=-1
)

# Run the search on the training set (already transformed by PCA)
grid_search.fit(X_train_pca_df, y_train_encoded)

# Get the best hyperparameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best hyperparameters:", best_params)
print("Best mean macro F1-score on cross-validation: {:.4f}".format(best_score))


# ### Training the model with the best hyperparameters

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import Pipeline as imbpipeline

# Define the best hyperparameters found for Random Forest
best_rf_params = {
    'max_depth': 10,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 200,
    'random_state': 42,
    'criterion': 'entropy'
}

# Define the classifier with the best hyperparameters
rf_model = RandomForestClassifier(**best_rf_params)

# Define the undersampling technique
cluster_centroids = ClusterCentroids(sampling_strategy='auto', random_state=42)

# Create a pipeline that first balances the classes and then trains the classifier
final_pipeline = imbpipeline([
    ('sampler', cluster_centroids),
    ('classifier', rf_model)
])

# Train the final model using the training set (already PCA-transformed)
final_pipeline.fit(X_train_pca_df, y_train_encoded)

print("Final model trained successfully with best hyperparameters and class balancing.")


# ### Final Model Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Use the best model to predict on test set
y_pred_rf_cluster = final_pipeline.predict(X_test_pca_df)
y_proba_rf_cluster = final_pipeline.predict_proba(X_test_pca_df)[:, 1]

# Set Seaborn style and Times New Roman font
sns.set(style="darkgrid")
plt.rcParams['font.family'] = 'Times New Roman'

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred_rf_cluster)
class_labels = ['BROKEN', 'NORMAL']

# Format confusion matrix with counts and percentages
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
group_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
group_percentages_str = ["{:.2%}".format(value) for value in group_percentages.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages_str)]
annot_labels = np.array(labels).reshape(2, 2)

# Setup figure
fig, axes = plt.subplots(figsize=(12, 5), ncols=2)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=annot_labels, fmt='', cmap='Blues', annot_kws={"size": 14}, ax=axes[0],
            xticklabels=class_labels, yticklabels=class_labels)
axes[0].set_title('Confusion Matrix (Test Set)', fontsize=16)
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('True Label')
axes[0].text(0.5, -0.2, '(a)', size=14, ha='center', transform=axes[0].transAxes)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_proba_rf_cluster)
roc_auc = roc_auc_score(y_test_encoded, y_proba_rf_cluster)

axes[1].plot(fpr, tpr, color='dodgerblue', lw=2,
             label='ROC curve (AUC = {:.2f})'.format(roc_auc))
axes[1].plot([0, 1], [0, 1], color='black', lw=0.6, linestyle='--')
axes[1].set_xlim([-0.02, 1.0])
axes[1].set_ylim([0.0, 1.02])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC)')
axes[1].legend(loc="lower right")
axes[1].text(0.5, -0.2, '(b)', size=14, ha='center', transform=axes[1].transAxes)

# Layout adjustments
plt.tight_layout()
plt.savefig('model_evaluation_results.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

# Classification Report
class_report = classification_report(y_test_encoded, y_pred_rf_cluster, target_names=class_labels)
print("\nClassification Report:")
print(class_report)

# Accuracy Score
accuracy = accuracy_score(y_test_encoded, y_pred_rf_cluster)
print(f"\nAccuracy: {accuracy:.4f}")


# # Importance Analysis with SHAP Values

# In[ ]:


# Initialize JavaScript visualization for SHAP interactive plots
shap.initjs()


# In[ ]:


import shap

# Set Seaborn's style
sns.set(style="darkgrid")

# Configure the font
plt.rcParams['font.family'] = 'Times New Roman'

# Remove timestamp column 
X_train_features = X_train.drop(columns=['timestamp'])

# Access the Random Forest model from the pipeline
rf_model = final_pipeline.named_steps['classifier']

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_features)

# Plot Summary Plot
shap.summary_plot(shap_values[0], X_train_features, show=False)

plt.savefig('Shap_values.png', format='png', dpi=600, bbox_inches='tight')
plt.show()


# ### SHAP Dependence Plot for sensor_04

# In[ ]:


import shap
import matplotlib.pyplot as plt

# SHAP dependency plot
shap.dependence_plot("sensor_04", shap_values[0], X_train_features, interaction_index=None)


# # False positive analysis

# ### Creating a dataset with false positives¶

# In[ ]:


# Convert numpy arrays to pandas DataFrames
X_test_df = pd.DataFrame(X_test, columns=X_test.columns)  
y_test_df = pd.DataFrame(y_test)
y_pred_rf_cluster_df = pd.DataFrame(y_pred_rf_cluster, columns=['y_expected_rf_cc'])

# Reset DataFrame indexes
X_test_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_rf_cluster_df.reset_index(drop=True, inplace=True)

# Join the DataFrames
dataset = pd.concat([X_test_df, y_test_df, y_pred_rf_cluster_df], axis=1)
dataset['machine_status'] = dataset['machine_status'].replace({'BROKEN': 0, 'NORMAL': 1})

# Export dataset
#dataset.to_excel('dataset_false_positive_analysis.xlsx', index=False)

# Create a new DataFrame containing only false positives
FP_dataset = dataset.loc[dataset['machine_status'] != dataset['y_expected_rf_cc']]

# Delete machine_status and y_expected_rf_cc
FP_dataset = FP_dataset.drop(['machine_status', 'y_expected_rf_cc', 'timestamp'], axis=1)


# ### Analyze the importance of false positive features

# In[ ]:


import pandas as pd

import shap

explainer_fp = shap.TreeExplainer(rf_model)
shap_values_fp = explainer.shap_values(FP_dataset)

# visualize the first prediction's explanation 
shap.initjs()
shap.force_plot(explainer_fp.expected_value[0], shap_values_fp[0][0,:], FP_dataset.iloc[0, :], matplotlib=True)


# In[ ]:


# Set Seaborn style after font setup
sns.set(style="darkgrid")

# Set the font before setting the Seaborn style
plt.rcParams['font.family'] = 'Times New Roman'

# Plot Summary Plot
shap.summary_plot(shap_values_fp[0], FP_dataset, show=False)

#plt.savefig('Most_important_feature_FP_dataset.png', format='png', dpi=600, bbox_inches='tight')
plt.show()


# ### SHAP dependency plot for sensor_04 from the false positive dataset

# In[ ]:


import shap
import matplotlib.pyplot as plt

# SHAP dependency plot
shap.dependence_plot("sensor_04", shap_values[0], FP_dataset, interaction_index=None)


# ### SHAP Analysis for False Positives

# In[ ]:


import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set font and style
plt.rcParams['font.family'] = 'Times New Roman'
sns.set(style="darkgrid")

# Make sure the model is extracted from the pipeline
rf_model = final_pipeline.named_steps['classifier']

# Create the explainer using the trained model
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values only for false positives
shap_values_fp = explainer.shap_values(fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore'))

# Plot global feature importance (BROKEN class)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_fp[1],  # SHAP values for positive class ('BROKEN')
    fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore'),
    plot_type="bar",
    show=False
)
plt.title("SHAP Mean Absolute Value of SHAP Values (False Positives)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_barplot_false_positives.png', dpi=600, bbox_inches='tight')
plt.show()

# Detailed feature effects plot
plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_fp[1],  # SHAP values for positive class ('BROKEN')
    fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore'),
    show=False
)
plt.title("SHAP Summary Plot - Feature Effects on False Positives", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_scatterplot_false_positives.png', dpi=600, bbox_inches='tight')
plt.show()


# ### Dependence.plot analysis of false positives from sensor_04

# In[ ]:


import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set font and style
plt.rcParams['font.family'] = 'Times New Roman'
sns.set(style="darkgrid")

# Step 1: Make sure the model is extracted from the pipeline
rf_model = final_pipeline.named_steps['classifier']

# Step 2: Recreate the explainer using the trained model
explainer = shap.TreeExplainer(rf_model)

# Step 3: Calculate SHAP values for false positives only
shap_values_fp = explainer.shap_values(
    fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore')
)

# Step 4: Plot dependence plot for sensor_04 (BROKEN class - index 1)
plt.figure(figsize=(8, 6))
shap.dependence_plot(
    "sensor_04",  # Feature of interest
    shap_values_fp[1],  # SHAP values for BROKEN class
    fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore'),
    interaction_index=None,  # Auto-detect most important interaction
    display_features=fp_dataset.drop(columns=['true_status', 'predicted_status', 'timestamp'], errors='ignore'),
    show=False
)

# Step 5: Add title and save figure
plt.title("SHAP Dependence Plot - sensor_04 (False Positives)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_dependence_sensor_04_false_positives.png', dpi=600, bbox_inches='tight')
plt.show()


# ### Scarter plot

# In[ ]:


import matplotlib.pyplot as plt
# Configurar a fonte
plt.rcParams['font.family'] = 'Times New Roman'

# Plotando os gráficos
for column in FP_dataset.columns[:-1]:  # Itera sobre todas as colunas, exceto a última (que é a coluna alvo)
    plt.figure(figsize=(18, 2.5))  # Ajuste o valor da altura conforme necessário
    
    # Plotando para o estado normal (0)
    plt.plot(dataset.loc[dataset['y_expected_rf_cc'] == 1, column], 
             label='NORMAL', linestyle='none', marker='o', markersize=1)
    
    # Plotando para o estado quebrado (1)
    plt.plot(dataset.loc[dataset['y_expected_rf_cc'] == 0, column], 
             label='BROKEN', linestyle='none', marker='X', color='red', markersize=1)
    
    plt.title(column)
    plt.xlabel('Timestamp (min)', fontsize=13)
    plt.ylabel('Valores', fontsize=13)
    plt.legend(prop={'size': 10})
    
    plt.xticks(fontsize=14)  # Ajuste o valor da fonte conforme necessário
    plt.yticks(fontsize=14)
    
    plt.show()

