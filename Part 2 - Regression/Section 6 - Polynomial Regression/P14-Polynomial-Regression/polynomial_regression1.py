#Polynomial Regression

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
#X=dataset.iloc[:,1] # it will load X as a vector but we would want to load as a 
               # matrix alwsa 
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Spliting the dataset in Training set and Test set
'''from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)'''


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""               

# Fitting Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualizing the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue',marker='+',markerfacecolor='g')
plt.title('Truth or Bluf(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue',marker='+',markerfacecolor='g')
plt.title('Truth or Bluf(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# To make curve more smooth 
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue',marker='+',markerfacecolor='g')
plt.title('Truth or Bluf(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new results with Linear Regression 
lin_reg.predict(np.array([6.5]).reshape(-1,1))
#Predicting a new results with Polynomial Regression 
lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(-1-1)))

