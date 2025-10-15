import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("./data/training_data/mixed_demographics.csv")

X = data.iloc[:, 2:8]

X_scaled = StandardScaler().fit_transform(X) #subtract mean, divide by standard deviation
pca = PCA(n_components = 6) #question says we want 6 components
scores = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,              # transpose so rows = variables
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns
)


print(loadings)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)
print(cum_explained)
#3 princiapl components explain 89% of the variation, 4 explains 96%. We will keep 4

pc_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])]) #convert data into PCs

pc_df_MI = pc_df


pc_df_MI["MI"] = data["MI"] #add the MI column back in

sns.pairplot(pc_df_MI, hue = "MI")
plt.show()


pc_df_sex = pc_df


pc_df_sex["sex"] = data["sex"] #add the sex column back in

sns.pairplot(pc_df_sex, hue = "sex")
plt.show()