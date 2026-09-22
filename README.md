# Simple Linear Regression

An end-to-end implementation of a **Simple Linear Regression** model using `scikit-learn` to predict employee salary based on years of experience.

---

## 📌 Machine Learning Pipeline

1. **Data Collection & Load:** Load dataset from source CSV (`Salary_dataset.csv`).
2. **Data Preparation & Cleaning:** Identify and drop unneeded/unnamed columns, check for nulls, and verify duplicates.
3. **Data Splitting:** Divide dataset into feature vector (`x`) and target vector (`y`), followed by a train-test split.
4. **Model Building & Fitting:** Reshape input features and fit `LinearRegression()` on training data.
5. **Evaluation:** Evaluate performance using $R^2$ score and regression metrics.

---

## 📊 Dataset Overview

- **Source:** `Salary_dataset.csv`
- **Shape:** 30 rows × 2 target variables
- **Features:** 
  - `YearsExperience` (Independent Feature $X$)
  - `Salary` (Dependent Target Variable $y$)

---

## 💻 Implementation & Code Workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset & Clean
ds = pd.read_csv(r"SML\Regression\Salary_dataset.csv")
ds.drop(columns=['Unnamed: 0'], inplace=True)  # Remove unnecessary index column

# 2. Exploratory Data Visualization
plt.scatter(ds['YearsExperience'], ds['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary')
plt.show()

# 3. Extract Features & Target
x = ds.iloc[:, 0]
y = ds.iloc[:, 1]

# 4. Train-Test Split (80% Train, 20% Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 5. Model Initialization & Training
lr = LinearRegression()
lr.fit(x_train.values.reshape(-1, 1), y_train)  # Reshape 1D array to 2D matrix

# 6. Prediction
model_predict = lr.predict(x_test.values.reshape(-1, 1))

# 7. Model Evaluation
r2 = r2_score(y_test, model_predict)
print(f"R² Score: {r2 * 100:.2f}%")
