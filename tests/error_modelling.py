# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
data = np.load('./data/labels_data.npy')
data_gt = np.load('./data/labels_gt_data.npy')

# Compute error
error = data_gt - data

# Stack together
data = np.hstack((data_gt, error))

# To pandas
columns=['X', 'Y', 'Z', 'ex', 'ey', 'ez']
df = pd.DataFrame(data, columns=columns)
print(df.describe())

# Plot
sns.pairplot(df[columns])
#plt.show()

# Model using linear regression
model = LinearRegression()
model.fit(df[['X', 'Y', 'Z']], df[['ex', 'ey', 'ez']])

# Compute score
r_sq = model.score(df[['X', 'Y', 'Z']], df[['ex', 'ey', 'ez']])

# Print results
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
print(r_sq)

# Predict 
y_pred = model.intercept_.reshape(3,1) + np.matmul(model.coef_, np.array([[600], [250], [5]]))
print(y_pred)
print(model.predict(np.array([600, 250, 5]).reshape(1, -1)))