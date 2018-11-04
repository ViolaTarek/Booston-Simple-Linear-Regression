"""
Spyder Editor
This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for crim column
data = pd.read_csv('Boston.csv')
x = data["crim"]
y = data["medv"]
meanx = np.mean(x)
meany= np.mean(y)
B1 = sum((x - meanx) * (y - meany)) / sum((x - meanx)**2) 
B0 = meany - B1*meanx
Ypredicted= B0 + B1 * x
print("Y predicted form the boston datase \n",Ypredicted)
#plot
plt.plot(x,y,'mo')
plt.xlabel('crim')
plt.ylabel('medv')
fit = np.polyfit(x,Ypredicted,1)
fit_fn = np.poly1d(fit) 
plt.plot(x,Ypredicted, 'co', x, fit_fn(x), '--k')
plt.show()
#Read and implement linear regression for Black Column
x2 = data["black"]
y2 = data["medv"]
meanx2 = np.mean(x2)
meany2= np.mean(y2)
B12 = sum((x2 - meanx2) * (y2 - meany2)) / sum((x2 - meanx2)**2) 
B02 = meany2 - B12*meanx2
Ypredicted2= B02 + B12 * x2
print("Y predicted form the boston datase black feature \n",Ypredicted2)
plt.plot(x2,y2,'co')
plt.xlabel('Black')
plt.ylabel('medv')
fit2 = np.polyfit(x2,Ypredicted2,1)
fit_fn2 = np.poly1d(fit2) 
plt.plot(x2,Ypredicted2, 'ro', x2, fit_fn2(x2), '--k')
plt.show()
