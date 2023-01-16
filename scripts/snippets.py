import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

def snippet_01():
    ###################################
    # Generate Data 
    ###################################

    sd = np.random.choice(range(0,2000))
    sd = 1956
    np.random.seed(sd)
    #print("Seed:", sd)

    N = 10
    x = np.linspace(2,8,N)
    y =  16 - (x - 6.5)**2 + np.random.normal(0, 1, N)

    xgrid = np.linspace(0,10,50)

    ###################################
    # Linear Model
    ###################################
    from sklearn.linear_model import LinearRegression

    m1 = LinearRegression()
    m1.fit(x.reshape(N,1),y)
    pred1 = m1.predict(xgrid.reshape(50,1))

    ###################################
    # Quadratic Model
    ###################################
    from sklearn.preprocessing import PolynomialFeatures

    pt = PolynomialFeatures(2)
    Xpoly = pt.fit_transform(x.reshape(N,1))
    m2 = LinearRegression()
    m2.fit(Xpoly,y)
    pred2 = m2.predict(pt.transform(xgrid.reshape(50,1)))


    ###################################
    # Degree 10 Model
    ###################################
    from sklearn.preprocessing import PolynomialFeatures

    pt = PolynomialFeatures(10)
    Xpoly = pt.fit_transform(x.reshape(N,1))
    m3 = LinearRegression()
    m3.fit(Xpoly,y)
    pred3 = m3.predict(pt.transform(xgrid.reshape(50,1)))

    ###################################
    # Piecewise Linear
    ###################################
    from scipy.optimize import minimize

    b = [6,-9,4.5,-3]

    def pred(b, x0):
        p1 = b[1] + b[2]*x0
        p2 = b[1] + b[0]*(b[2] - b[3]) + b[3]*x0
        p = np.where(x0 < b[0], p1, p2)
        return p

    def sse(b):    
        p = pred(b, x)
        e = p - y
        return np.sum(e**2)

    min_results = minimize(sse, b)
    b_opt = min_results.x
    pred4 = pred(b_opt, xgrid)

    ###################################
    # KNN Model
    ###################################
    from sklearn.neighbors import KNeighborsRegressor

    m5 = KNeighborsRegressor(3)
    m5.fit(x.reshape(N,1),y)
    pred5 = m5.predict(xgrid.reshape(50,1))

    ###################################
    # Plot
    ###################################

    x0 = 0
    x1 = 10
    y0 = 0
    y1 = 20

    plt.close()
    plt.rcParams["figure.figsize"] = [12,8]

    plt.subplot(2,3,1)
    plt.scatter(x, y)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Original Data')

    plt.subplot(2,3,2)
    plt.plot(xgrid, pred1, c='darkorange', zorder=1)
    plt.scatter(x, y, zorder=2)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Model 1: Linear Regression')

    plt.subplot(2,3,3)
    plt.plot(xgrid, pred2, c='darkorange', zorder=1)
    plt.scatter(x, y, zorder=2)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Model 2: Quadratic Model')

    plt.subplot(2,3,4)
    plt.plot(xgrid, pred3, c='darkorange', zorder=1)
    plt.scatter(x, y, zorder=2)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Model 3: Degree 10 Poly')

    plt.subplot(2,3,5)
    plt.plot(xgrid, pred4, c='darkorange', zorder=1)
    plt.scatter(x, y, zorder=2)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Model 4: PW-Linear Regression')

    plt.subplot(2,3,6)
    plt.plot(xgrid, pred5, c='darkorange', zorder=1)
    plt.scatter(x, y, zorder=2)
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.title('Model 5: 3-Nearest Neighbors')

    plt.show()
