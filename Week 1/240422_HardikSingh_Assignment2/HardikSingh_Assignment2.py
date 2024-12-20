import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class LinearReg():
    def __init__(self, lr = 0.00325, n_iters=5000):
        self.lr=lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            ypred = np.dot(X, self.weights) + self.bias

            dw = (2/n_samples) * np.dot(X.T, (ypred-y))
            db = (2/n_samples) * np.sum(ypred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr*db

            # if i % 50 == 0:
                # print(f"Iteration {i}: Weights={self.weights}, Bias={self.bias}, Loss={np.mean((y - ypred) ** 2)}")
        
    def predict(self, X):
        ypred = np.dot(X, self.weights) + self.bias
        return ypred

df = pd.read_csv('housing.csv')
# print(df)

df.replace({'yes':1, 'no':0, 'furnished':2, 'semi-furnished':1, 'unfurnished':0}, inplace=True)
X = df.drop('price', axis=1)
X['area'] = X['area']/1e4
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=123)


y_train = y_train/1e6
y_test = y_test/1e6

reg = LinearReg()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

def MSE(pred, actual):
    return np.mean((pred-actual)**2)

mse = MSE(pred, y_test)

# print(pred)
# print(np.abs(pred-y_test))

print(mse) # This is the accuracy. Note that the mean squared error is in terms of millions
