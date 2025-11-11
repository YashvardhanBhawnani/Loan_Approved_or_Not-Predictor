# Loan Approval Prediction (Machine Learning Project)
---
## 🎯 Objective

The goal of this project is to build and evaluate predictive models to determine whether a loan application will be approved. The models are trained on a real-world dataset (`loan_approval.csv`) containing applicant financial data.

The project demonstrates a complete machine learning pipeline, including:
* Data Loading & Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering & Scaling
* Model Training (Logistic Regression, Random Forest)
* Model Validation (5-Fold Cross-Validation)
* Hyperparameter Tuning (`GridSearchCV`)
* Final Model Evaluation & Comparison

---

## 📁 Dataset

The project uses the `loan_approval.csv` dataset, which contains 2000 rows of applicant data.

### Key Features:
* **`income`**: Applicant's annual income
* **`credit_score`**: Applicant's credit score
* **`loan_amount`**: The amount of money requested
* **`years_employed`**: Number of years employed
* **`loan_approved`**: (Target Variable) 1 = Approved, 0 = Denied

---

## 🛠️ Project Workflow

1.  **Data Loading:** The `loan_approval.csv` file is imported into a Pandas DataFrame.
2.  **Preprocessing:**
    * Non-numeric columns (`name`, `city`) are dropped.
    * Invalid data (e.g., 0s in `loan_amount`) is replaced with `NaN`.
    * All missing `NaN` values are imputed using the **median** of their respective columns.
3.  **Exploratory Data Analysis (EDA):**
    * **Histograms** are plotted to visualize feature distributions.
    * A **Correlation Heatmap** is generated to find relationships between features and the target.
    * **Boxplots** are used to identify outliers.
4.  **Feature Engineering & Scaling:**
    * **Critical Finding:** The `points` column was identified as a **"data leaker"**. It had an extremely high correlation (~0.82) with the target, causing an unrealistic 100% accuracy. This feature was **removed** from the dataset before training to build a valid model.
    * All remaining features are standardized using `StandardScaler`.
5.  **Model Building & Validation:**
    * The data is split into an 80% training set and a 20% test set.
    * **Model 1: Logistic Regression** is trained as a baseline.
    * **Model 2: Simple Random Forest** is trained with default parameters.
    * **5-Fold Cross-Validation** is run on the training set for both base models to get a stable, reliable accuracy estimate.
6.  **Model Tuning:**
    * **Model 3: Tuned Random Forest** is created using `GridSearchCV` (with `cv=5`) to find the optimal hyperparameters.
7.  **Evaluation:**
    * All three models are evaluated on the held-out test set.
    * Performance is compared using Accuracy, F1-Score, Classification Reports, Confusion Matrices, and ROC/AUC curves.

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (sklearn)

---

## ⚡ How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd YOUR_REPOSITORY
    ```
3.  Install the required libraries. (It's recommended to create a `requirements.txt` file).
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  Run the main Python script (e.g., `loan_approval.py`) or open the Jupyter Notebook.
    ```bash
    python your_script_name.py
    ```
    *(This will run the full pipeline, print the results to the console, and save all plots as `.png` files in the directory.)*

---

## 📊 Results & Evaluation

After removing the leaky `points` feature, the models achieved the following realistic performance on the 20% test set:

| Model | Test Set Accuracy | 5-Fold CV Mean (on Train) |
| :--- | :---: | :---: |
| Logistic Regression | 92.5% | 92.3% |
| Simple Random Forest | **97.5%** | 97.2% |
| Tuned Random Forest | **97.5%** | (Tuned via CV) |

### Key Observations:
* The **Random Forest Classifier (97.5% accuracy)** significantly outperformed the Logistic Regression model. This indicates the data has complex, non-linear relationships that the ensemble model could capture.
* The **Simple and Tuned Random Forest models had identical performance.** This is a positive result, showing that the default hyperparameters for Random Forest were already optimal for this dataset.
* All evaluation plots (Confusion Matrices, ROC Curves, and the final accuracy comparison chart) are saved to the repository.

---

## 🏁 Conclusion

This project successfully demonstrates the creation of a complete machine learning pipeline for financial risk assessment. The **Tuned Random Forest Classifier** was the best-performing model, achieving **97.5% accuracy** in predicting loan approvals.

A critical takeaway was the identification and removal of a "leaky" feature (`points`), which is a crucial step in building a valid, generalizable, and realistic predictive model.
