import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Perceptron(object):

    def __init__(self,input_size, lr=1 , epoch=10):
        self.W = np.zeros(input_size+1)
        self.epoch = epoch
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0


    def predict(self,x):
        z = self.W.T.dot(x)
        a = self.activation(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epoch):
            for i in range(d.shape[0]):
                x = np.insert(X[i] ,0,1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x



if __name__ == '__main__':
        X = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
        d = np.array([0,0,0,1])

        perceptron = Perceptron(input_size =2)
        perceptron.fit(X,d)
        print(perceptron.W)
        X0 = np.array([[0,0],[0,1],[1,0]])
        X00 =np.array([[1,1]])


        x1 = X[:,0]

        x01 = X0[:,0]
        x02 = X0[:, 1]
        x001 = X00[:, 0]
        x002 = X00[:, 1]
        print(x01)

        plt.title('The AND gate', fontsize= 20)
        plt.axis([-1,4,-1,4])
        plt.scatter(x01,x02,c='r', label='Class 1')
        plt.scatter(x001, x002, c='b',label='Class 2')
        plt.xlabel('x1')
        plt.ylabel('x2')


        xx =[]
        yy =[]
        for i in  x1:
            slope = -(perceptron.W[0]/perceptron.W[2]) / (perceptron.W[0]/perceptron.W[1])
            intercept = -perceptron.W[0]/perceptron.W[2]
            y = (slope*i) + intercept
            xx.append(i)
            yy.append(y)

        # adding to extra points for scaling the line
        xx.append(2)
        yy.append(-1)
        plt.plot(xx, yy,c='y', label='Decision Boundary')
        plt.legend()

        plt.show()