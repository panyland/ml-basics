import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

matplotlib.use('TkAgg')

# Define coefficients a and b (y = ax + b) 
def my_linfit(x, y):
    a = ( len(x)*np.sum(x*y) - np.sum(y)*np.sum(x) )/ ( len(x)*np.sum(x**2)-np.sum(x)**2 )
    b = ( np.sum(y)*np.sum(x**2) - np.sum(x)*np.sum(x*y) )/ ( len(x)*np.sum(x**2)-np.sum(x)**2 ) 
    return a, b

plt.figure()
plt.title('Click to select 10 points.')
plt.xlim(0, 100)
plt.ylim(0, 100)

# Ask for 10 points on the graph
points = plt.ginput(10, timeout=0)
x, y = zip(*points)
a, b = my_linfit(np.array(x), np.array(y))
plt.plot(x, y, 'kx')
xp = np.arange(0, 100, 0.1)
# Plot the line
plt.plot(xp, a*xp + b, 'r-')
plt.show()
