import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv("./data/training_data/mixed_demographics.csv")

X = data.iloc[:, 2:8]
y = data.iloc[:, 1]

y_binary = y_binary = (y == "pMI").astype(int)
y = y_binary #convert to 0-1

X_scaled = StandardScaler().fit_transform(X) #subtract mean, divide by standard deviation
pca = PCA(n_components = 6) #question says we want 6 components
scores = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,              # transpose so rows = variables
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns
)


print(loadings)

#PC associations with demographic features:

#PC1 - mostly with BMI, weight, diastolic and systolic BP as a weighted average
#PC2 - weighted difference between age, height and diastolic BP
#PC3 - weighted difference between age, systolic BP and height
#PC4 - BMI, diastolic and systolic



explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)
print(cum_explained)
#3 princiapl components explain 89% of the variation, 4 explains 96%. We will keep 4

pc_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])]) #convert data into PCs

pc_df_MI = pc_df

sns.pairplot(pc_df)
plt.show()


pc_df_MI["MI"] = data["MI"] #add the MI column back in

sns.pairplot(pc_df_MI, hue = "MI")
plt.show()


pc_df_sex = pc_df


pc_df_sex["sex"] = data["sex"] #add the sex column back in

sns.pairplot(pc_df_sex, hue = "sex")
plt.show()


#### LDA ####


# Fit LDA for projection (same API; transform returns up to n_classes-1 dims)
scaler = StandardScaler().fit(X)
lda_proj = LinearDiscriminantAnalysis(n_components = 1)
X_lda = lda_proj.fit_transform(scaler.transform(X), y)   # use full data for visualization

# Plot

X_lda_flat = np.ravel(X_lda)
plt.figure(figsize=(8,4))
scatter = plt.scatter(
    np.arange(len(X_lda_flat)), X_lda_flat,
    c=y, cmap='bwr', alpha=0.8, label=y
)
plt.title("LDA projection (LDA1 vs. Index)")
plt.xlabel("Sample index")
plt.ylabel("LDA component 1")
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, [f"Class {lab}" for lab in np.unique(y)], title="Classes")
plt.tight_layout()
plt.show()


