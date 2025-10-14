import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

data = pd.read_csv("./data/training_data/mixed_demographics.csv")

data.info()
data.describe()



#### 2a ####


sns.boxplot(y = "age", data = data)

sns.boxplot(x = "MI", y = "age", data = data)

#Results: Age is telling. Generally, healthy people are
#younger than people who have MI. Healthy median age ~ 63
#and MI median age ~ 65



plt.show()

sns.boxplot(y = "BMI", data = data)

sns.boxplot(x = "MI", y = "BMI", data = data)


#Results: BMI is not very telling on whether someone
#has an MI or not. No differences

plt.show()

sns.boxplot(y = "height", data = data)

sns.boxplot(x = "MI", y = "height", data=data)

#Results: Height is not telling on whether someone
#has an MI or not. No differences

plt.show()

sns.boxplot(y = "weight", data = data)

sns.boxplot(x = "MI", y = "weight", data=data)

#Results: Weight is not telling on whether someone
#has an MI or not. No differences

plt.show()

sns.boxplot(y = "diastolic_BP", data = data)

sns.boxplot(x = "MI", y = "diastolic_BP", data = data)

#Results: Diastolic_BP is not telling on whether someone
#has an MI or not. No differences

plt.show()

sns.boxplot(y = "systolic_BP", data = data)

sns.boxplot(x = "MI", y = "systolic_BP", data = data)

#Results: Systolic_BP is not telling on whether someone
#has an MI or not. No differences

plt.show()


mosaic(data, ["sex", "MI"])
plt.title("Mosaic Plot: Sex vs MI")
plt.show()

#Results: Sex is a good covariate in determing whether someone is
#healthy or not. Sex TRUE has a higher proportion of pMI than
#sex FALSE.

#(sex TRUE is Male, sex FALSE is Female)




#### 2b ####

#Evaulating correlations between covariates

subset = data.iloc[:, 1:8] #removes index number, MI and sex
sns.pairplot(subset)

plt.show()


#Results: All covariates are normally distributed, weight is
#more condensed around the centre than the others.

#Age correlated with height and weight mostly. Slightly with
#diastolic and systolic BP.

#BMI correlated with weight, diastolic and systolic. Makes sense
#due to the formula

#height correlated with weight

#weight correlated with diastolic and systolic

#diastolic correlated with systolic





#Plots to compare sex with other covariates

sns.countplot(x = "sex", data = data)
plt.show() #univariate plot 

#Remember, TRUE is male, FALSE is female. We have around 500 males and 350 females



sns.boxplot(x = "sex", y = "age", data = data)
plt.show() 

#Males in this study are generally older than females

sns.boxplot(x = "sex", y = "BMI", data = data)
plt.show()

#BMIs are somewhat equivalent across the sexes

sns.boxplot(x = "sex", y = "height", data = data)
plt.show()

#Generally males (median 65) are taller than females (median 62), which is expected


sns.boxplot(x = "sex", y = "weight", data = data)
plt.show()

#men are heavier than females, which again is expected

sns.boxplot(x = "sex", y = "diastolic_BP", data = data)
plt.show()

#diastolic_BP similar between the sexes

sns.boxplot(x = "sex", y = "systolic_BP", data = data)
plt.show()

#systolic_BP similar, however narrower IQR for females and median for females is smaller