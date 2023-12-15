# Imports
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
data_meas = np.load('./data/labels_data.npy')
data_gt = np.load('./data/labels_gt_data.npy')

# Compute error
error = data_meas - data_gt

# Stack together
data = np.hstack((data_gt, error))

# To pandas
columns=['X', 'Y', 'Z', 'ex', 'ey', 'ez']
df = pd.DataFrame(data, columns=columns)
print(df.describe())

# Plot
sns.pairplot(df[columns])
plt.show()

# Model using linear regression
model = LinearRegression()
model.fit(df[['X', 'Y', 'Z']], df[['ex', 'ey', 'ez']])

# Save model
#filename = './data/error_model.sav'
#pickle.dump(model, open(filename, 'wb'))

# Compute score
r_sq = model.score(df[['X', 'Y', 'Z']], df[['ex', 'ey', 'ez']])
print(r_sq)

# Corrected data
data_corr = data_meas - model.predict(df[['X', 'Y', 'Z']])

# Error
error = data_corr - data_gt

# Stack together
data = np.hstack((data_gt, error))

# To pandas
columns=['X', 'Y', 'Z', 'ex', 'ey', 'ez']
df = pd.DataFrame(data, columns=columns)
print(df.describe())

# Plot
sns.pairplot(df[columns])
plt.show()

# Plot data
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(data_meas[:,0], data_meas[:,1], data_meas[:,2], 'g*') # Plot data
ax.scatter(data_gt[:,0], data_gt[:,1], data_gt[:,2], 'r*') # Plot ground truth
ax.scatter(data_gt[:,0], data_gt[:,1], np.linalg.norm(error[:, 0:2], axis=1)*10, 'g*') # Plot error
ax.scatter(0, 0, 0, 'k*') # Plot robot 
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_aspect('equal')
#plot_frame_t(T_bc, ax) # Plot camera frame
#robot.plot(ax, np.array([0, 0, 0, 0, 0, 0])) # Plot robot in home pose
plt.show()