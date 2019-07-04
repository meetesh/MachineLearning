import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
def main():
 #get the input from file
 data = genfromtxt('ex1data1.txt',delimiter=',')
 X = data[:,0]
 y = data[:,1]
 m = X.size
 plotData(X,y)
 return

def plotData(x,y):
 plt.plot(x,y,'bx',linewidth=0)
 plt.show()
 return

if __name__ == '__main__':
 main() 
