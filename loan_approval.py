import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, roc_curve, roc_auc_score
)
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
db = pd.read_csv("loan_approval.csv")
print("\n Dataset Loaded Successfully!\n")

pd.set_option('display.max_columns', None)
print("First 10 Rows:")
print(db.head(10))

print("\n Shape of Dataset:", db.shape)

print("\n Dataset Information:")
print(db.info())
print("\n Summary Statistics:")
print(db.describe())

warnings.filterwarnings('ignore')

# STEP 1: DATA LOADING & PREPROCESSING
# Load your dataset
try:
    # Use a relative path (assumes file is in the same folder)
    db = pd.read_csv("loan_approval.csv")
except FileNotFoundError:
    print("Error: loan_approval.csv not found.")
    print("Please make sure the file is in the same directory as your script.")
    exit()

print("--- Initial Data Inspection ---")
print(db.head())

# Prepare Data for Analysis
db_analysis = db.drop(['name', 'city'], axis=1)
db_analysis['loan_approved'] = db_analysis['loan_approved'].astype(int)

# Check for missing or zero values
print("\nChecking for Missing Values (Initial):")
print(db_analysis.isnull().sum())
columns_with_zero = ['loan_amount']
db_analysis[columns_with_zero] = db_analysis[columns_with_zero].replace(0, np.nan)

# Handle Missing Values (Imputation)
print("\nHandling Missing Values (Imputation)...")
numeric_cols = db_analysis.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if db_analysis[col].isnull().sum() > 0:
        db_analysis[col].fillna(db_analysis[col].median(), inplace=True)

print("\nMissing values handled successfully!")


# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & SCALING

# Univariate Analysis – Histograms
print("\nPlotting Distributions...")
try:
    db_analysis.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Feature Distributions (Loan Approval)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('loan_feature_distributions.png')
    print("Saved feature distributions plot to 'loan_feature_distributions.png'")
    plt.show()
except Exception as e:
    print(f"Error plotting histograms: {e}")

# Correlation Heatmap
print("\nCorrelation Heatmap:")
plt.figure(figsize=(10, 7))
sns.heatmap(db_analysis.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('loan_correlation_heatmap.png')
print("Saved correlation heatmap to 'loan_correlation_heatmap.png'")
plt.show()

# Boxplots to detect outliers
print("\nVisualizing Outliers:")
plt.figure(figsize=(12, 8))
db_analysis.drop('loan_approved', axis=1).boxplot()
plt.title("Boxplot of Features", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('loan_feature_boxplots.png')
print("Saved boxplots to 'loan_feature_boxplots.png'")
plt.show()


# Feature Scaling (Standardization)
# Drop the target AND the leaky 'points' feature
X = db_analysis.drop(['loan_approved', 'points'], axis=1)
y = db_analysis['loan_approved']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (first 5 rows):")

# Create scaled dataframe for clarity
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("\n Scaled Data (first 5 rows):")
print(scaled_df.head())
print("\nFinal Dataset Ready for Model Building!")
print(f"Total Records: {db_analysis.shape[0]}")
print(f"Features: {len(X.columns)}")
print(f"Loans Approved: {y.sum()}")
print(f"Loans Denied: {len(db_analysis) - y.sum()}")
print("\n Preprocessing & Scaling Complete!")


# # STEP 3: MODEL BUILDING & COMPARISON
# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nData split completed!")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# STEP 3: MODEL BUILDING, VALIDATION & TUNING

# Model 1: Logistic Regression (Baseline)

print("\n--- Training Logistic Regression ---")
# Initialize the model
log_reg = LogisticRegression(max_iter=1000)

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = log_reg.predict(X_test)

# Calculate accuracy on the single test set split
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nLogistic Regression Accuracy (on Test Set): {round(acc_lr * 100, 2)}%")

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# Model 2: Random Forest Classifier (Simple)
print("\n--- Training Simple Random Forest ---")
# Initialize the model with a fixed random_state for reproducible results
rf = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Calculate accuracy on the single test set split
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nSimple Random Forest Accuracy (on Test Set): {round(acc_rf * 100, 2)}%")

print("\nClassification Report (Simple Random Forest):")
print(classification_report(y_test, y_pred_rf))


# Cross-Validation (5-fold) on Base Models
print("\n--- Performing 5-Fold Cross Validation (on Training Data) ---")

# Run 5-fold CV for Logistic Regression on the training set
cv_scores_lr = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')

# Run 5-fold CV for the simple Random Forest on the training set
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')

# Print the mean accuracy for both
print("Average CV Accuracy - Logistic Regression:", round(cv_scores_lr.mean() * 100, 2), "%")
print("Average CV Accuracy - Simple Random Forest:", round(cv_scores_rf.mean() * 100, 2), "%")


# Model 3: Tuned Random Forest (Hyperparameter Tuning)
print("\n--- Training & Tuning Random Forest (with 5-Fold CV) ---")
# Define the parameters to test
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2]
}

# Initialize a new model for tuning
rf_tuned = RandomForestClassifier(random_state=42)

# Set up GridSearchCV to test all parameter combinations
# It uses 5-fold cross-validation (cv=5) to find the best model
grid_search = GridSearchCV(estimator=rf_tuned, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model found by the grid search
best_rf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_best = best_rf.predict(X_test)

# Calculate the final accuracy
acc_best = accuracy_score(y_test, y_pred_best)

print("\nBest Random Forest Parameters Found:", grid_search.best_params_)
print(f"\nTuned Random Forest Accuracy (on Test Set): {round(acc_best * 100, 2)}%")

print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best))


# STEP 4: MODEL EVALUATION (METRICS & VISUALS)


# Logistic Regression Evaluation
print("\nLogistic Regression Evaluation:")
f1_lr = f1_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('confusion_matrix_lr.png')
print("Saved LR confusion matrix to 'confusion_matrix_lr.png'")
plt.show()

y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.savefig('roc_curve_lr.png')
print("Saved LR ROC curve to 'roc_curve_lr.png'")
plt.show()

# Tuned Random Forest Evaluation
print("\nTuned Random Forest Evaluation:")
f1_rf = f1_score(y_test, y_pred_best)
cm_rf = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix – Tuned Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('confusion_matrix_rf.png')
print("Saved RF confusion matrix to 'confusion_matrix_rf.png'")
plt.show()

y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
plt.figure(figsize=(6, 5))
plt.plot(fpr_rf, tpr_rf, label=f"Tuned Random Forest (AUC = {auc_rf:.3f})", color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Tuned Random Forest")
plt.legend()
plt.savefig('roc_curve_rf.png')
print("Saved RF ROC curve to 'roc_curve_rf.png'")
plt.show()

# Model Comparison
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Tuned Random Forest'],
    'Accuracy': [round(acc_lr * 100, 2), round(acc_best * 100, 2)],
    'F1-Score': [round(f1_lr, 3), round(f1_rf, 3)],
    'AUC': [round(auc_lr, 3), round(auc_rf, 3)]
})
print("\nModel Comparison Summary:")
print(comparison)

# Bar Chart Comparison
print("\nPlotting Model Comparison...")
plt.figure(figsize=(7, 5))
sns.barplot(x='Model', y='Accuracy', data=comparison, palette=['blue', 'green'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig('model_accuracy_comparison.png')
print("Saved model comparison bar chart to 'model_accuracy_comparison.png'")
plt.show()

print("\n--- Full Analysis Finished ---") 