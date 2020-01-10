# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Spliting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)

# Fitting the Decision Tree Regression Model  to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#Predicting a new results with SVR 
y_pred=regressor.predict(np.array([6.5]).reshape(-1,1))

#Visualizing the Decision Tree Regression Model results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluf(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# To make curve more smooth  SVR
#Visualizing the Decision Tree Regression Model results(for higher resolutio
 # smoother curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue',marker='+',markerfacecolor='g')
plt.title('Truth or Bluf( SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Decision Tree Regression Model results(for higher resolutio
 # smoother curve)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluf( SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



# Predicting a new results 
y_pred=regressor.predict(85)

# Visualizing the Decision Tree Regression results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue',marker='+')
plt.title('Truth or Bluf(Decision Tree Regression )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()