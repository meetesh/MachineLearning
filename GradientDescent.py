import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def computeCost(X,y,theta):
 m = y.size
 J = 0
 hypothesis = X @ theta 
 t = np.sum(((hypothesis-y))**2) 
 J = (1/(2*m)) * t
 return J

def gradientDescent(X,y,theta,alpha,iterations):
 m = y.shape[0] 
 for i in range(iterations):
  h = X @ theta
  # non vectorized Code
  #temp1 = theta[0] - (alpha * (1/m) * np.sum((h-y)*(X[:,0].reshape(m,1))))
  #temp2 = theta[1] - (alpha * (1/m) * np.sum((h-y)*(X[:,1].reshape(m,1))))
  #theta = np.vstack([temp1,temp2])
  # vectorized Code 
  theta = theta - (alpha * (1/m) * ((h-y).transpose() @ X).transpose());
 return theta

def main():
 #get the input from file
 data = genfromtxt('ex1data1.txt',delimiter=',')
 X = data[:,0]
 y = data[:,1].reshape(data.shape[0],1)
 m = X.size
 print("Plotting the data...")
 plotData(X,y)

 #add one extra column for X0 with value as 1
 X=np.column_stack((np.ones([m,1]),X))
 
 #creating paramenter array with values as 0
 theta = np.zeros([2,1])

 #testing the cost function for initial value of theta
 print("Testing the cost function....")
 J = computeCost(X,y,theta)
 print("The cost function return value is %f and expected is 32.07(approx)"%J)

 #try run iteration with gradient descent algorithm  
 iterations = 1500;
 alpha = 0.01;
 print("Running Gradient Descent.....")
 theta = gradientDescent(X, y, theta, alpha, iterations);
 print("Theta found by gradient descent is \n",theta);
 print("Expected value is -3.6303 and 1.1664")
 plt.plot(X[:,1],X @ theta)
 plotData(X[:,1],y)

 #plotting the cost computation graph
 theta0_vals = np.linspace(-10,10,100)
 theta1_vals = np.linspace(-1,4,100)
 J_vals = np.zeros([theta0_vals.size,theta1_vals.size])
 for i in range(theta0_vals.size):
  for j in range(theta1_vals.size):
   t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
   J_vals[i,j] = computeCost(X,y,t)
 J_vals = J_vals.transpose(); 
 fig = plt.figure();
 ax = fig.gca(projection='3d')
 surf = ax.plot_surface(theta0_vals.reshape(100), theta1_vals.reshape(100), J_vals.reshape(100,100), cmap=cm.coolwarm)
 plt.show();
 return

def plotData(x,y):
 plt.plot(x,y,'bx',linewidth=0)
 plt.show()
 return

if __name__ == '__main__':
 main() 
