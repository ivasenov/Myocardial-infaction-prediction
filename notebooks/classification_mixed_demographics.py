import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, auc, classification_report
import matplotlib.pyplot as plt


data = pd.read_csv("./data/training_data/mixed_demographics.csv")

X = data.iloc[:, 2:9]

#from EDA, we know sex, age are the best. BMI is good relative to its scale, systolic is good
#issue is that sex and age are correlated in this dataset, may need regularisation to account for this



y = data.iloc[:, 1]


y_binary = y_binary = (y == "pMI").astype(int)
y = y_binary #convert to 0-1








X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # fit on train only
X_test_std  = scaler.transform(X_test)       # transform test with same params

Cs = np.logspace(-6, 6, 100)
l1_ratios = np.linspace(0, 1, 21)  # ridge to lasso

cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)


lr_cv = LogisticRegressionCV(
    Cs = Cs,
    cv = cv,
    penalty = 'elasticnet',
    solver = 'saga',
    l1_ratios = l1_ratios,
    scoring = 'roc_auc',   # or 'neg_log_loss' or 'accuracy', pick what you want
    max_iter = 20000,
    tol = 1e-4,
    n_jobs = -1,
    refit = True          # refit on full data with best params
)


lr_cv.fit(X, y)
final_coefs = lr_cv.coef_

coef_df = pd.DataFrame({
    'variable': X.columns,
    'exp(coefficient)': np.exp(lr_cv.coef_[0])
})

print(coef_df)

print(1 / lr_cv.C_[0])
print(lr_cv.l1_ratio_)


y_pred = lr_cv.predict(X_test)
y_pred_prob = lr_cv.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC on test set: {auc:.4f}")

RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title(f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.3f})")
plt.show()

print(classification_report(y_test, y_pred, target_names=["Healthy", "pMI"]))


