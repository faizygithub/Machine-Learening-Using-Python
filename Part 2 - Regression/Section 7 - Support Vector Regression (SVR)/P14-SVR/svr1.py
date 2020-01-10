#SVR

#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Spliting the dataset in Training set and Test set
'''from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)'''


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(np.array(X).reshape(-1,1))
#X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.array(y).reshape(-1,1)) 

#Fitting SVR to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


#Predicting a new results with SVR 
#y_pred=regressor.predict(np.array([6.5]).reshape(-1,1))
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing the SVR results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue',marker='+')
plt.title('Truth or Bluf(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# To make curve more smooth  SVR
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue',marker='+',markerfacecolor='g')
plt.title('Truth or Bluf( SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()