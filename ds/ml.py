import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, MinMaxScaler, RobustScaler, Normalizer, SimpleImputer
from sklearn import linear_model

# MinMaxScaler cales each feature (column) of the data to a given range (where the default range is [0, 1]).
default_scaler = MinMaxScaler() # the default range is [0,1]
transformed = default_scaler.fit_transform(data)


#Data standardization uses each feature's mean and standard deviation, while ranged scaling uses the maximum and minimum feature values, meaning that they're both susceptible to being skewed by outlier values.
#robustly scale the data, i.e. avoid being affected by outliers, by using use the data's median and Interquartile Range (IQR). Since the median and IQR are percentile measurements of the data (50% for median, 25% to 75% for the IQR), they are not affected by outliers.
robust_scaler = RobustScaler()
transformed = robust_scaler.fit_transform(data)

#L2 normalization applied to a particular row of a data array will divide each value in that row by the row's L2 norm. In general terms, the L2 norm of a row is just the square root of the sum of squared values for the row.
normalizer = Normalizer()
transformed = normalizer.fit_transform(data)

#data imputation methods: 
# Using the mean value
# Using the median value
# Using the most frequent value
# Filling in missing values with a constant

imp_mean = SimpleImputer()
transformed = imp_mean.fit_transform(data)

imp_median = SimpleImputer(strategy='median')
transformed = imp_median.fit_transform(data)

imp_constant = SimpleImputer(strategy='constant',
                             fill_value=-1)
transformed = imp_constant.fit_transform(data)

#PCA
pca_obj = PCA(n_components=3)
pc = pca_obj.fit_transform(data).round(3)

#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(pizza_data, pizza_prices)

new_pizzas = np.array([[2000,  820],
                       [2200,  830]])
price_predicts = reg.predict(new_pizzas)
print('{}\n'.format(repr(price_predicts)))
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
r2 = reg.score(pizza_data, pizza_prices)
print('R2: {}\n'.format(r2))

#Ridge Regression
reg = linear_model.Ridge(alpha=0.1)
reg.fit(pizza_data, pizza_prices)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
r2 = reg.score(pizza_data, pizza_prices)
print('R2: {}\n'.format(r2))

alphas = [0.1, 0.2, 0.3]
reg = linear_model.RidgeCV(alphas=alphas)
reg.fit(pizza_data, pizza_prices)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
print('Chosen alpha: {}\n'.format(reg.alpha_))


reg = linear_model.Lasso(alpha=0.1)
reg.fit(data, labels)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
print('R2: {}\n'.format(reg.score(data, labels)))

#logistic regression because it performs regression on logits, which then allows us to classify the data based on model probability predictions.
reg = linear_model.LogisticRegression()
reg.fit(data, labels)

new_data = np.array([
  [  0.3,  0.5, -1.2,  1.4],
  [ -1.3,  1.8, -0.6, -8.2]])
print('Prediction classes: {}\n'.format(
  repr(reg.predict(new_data))))

reg = linear_model.LogisticRegression(
  solver='lbfgs',
  multi_class='multinomial', max_iter=200)
reg.fit(data, labels)

new_data = np.array([
  [ 1.8, -0.5, 6.2, 1.4],
  [ 3.3,  0.8, 0.1, 2.5]])
print('Prediction classes: {}\n'.format(
  repr(reg.predict(new_data))))