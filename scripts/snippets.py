import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize
import warnings 

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def plot_regions(model, X, y, num_ticks=100, cmap='rainbow', 
                 fig_size=None, legend=True, close=True, 
                 display=True, path=None, keras=False):

    # Convert X to numpy array
    X = np.array(X)
    y = np.array(y)
    
    # Check to see if there are exactly 2 features
    if X.shape[1] != 2:
        raise Exception('Training set must contain exactly two features.')
        
    # Find min and max points for grid axes
    minx, maxx = min(X[:,0]), max(X[:,0])
    marginx = (maxx - minx) / 20
    x0, x1 = minx - marginx, maxx + marginx
    
    miny, maxy = min(X[:,1]), max(X[:,1])
    marginy = (maxy - miny) / 20
    y0, y1 = miny - marginy, maxy + marginy
    
    # Create grid tick marks
    xticks = np.linspace(x0, x1, num_ticks)
    yticks = np.linspace(y0, y1, num_ticks)
    
    # Create array of grid points
    # tile stacks copies of xticks end creating a size num_ticks^2 array
    # repeat repeats each elt of yticks to create a size num_ticks^2 array
    # They are combined into an array of shape (2, num_ticks^2)
    # transpose creates array of pts with shape (num_ticks^2, 2).
    grid_pts = np.transpose([np.tile(xticks,len(yticks)),
                             np.repeat(yticks,len(xticks))])
    
    # Feed grid points to model to generate 1D array of classes
    if(keras==True): 
        class_pts = model.predict_classes(grid_pts)
        class_pts = class_pts.reshape((len(class_pts),))
    else:
        class_pts = model.predict(grid_pts)

    # Get list of classes. This could, in theory, contain text labels.
    classes = np.unique(y)
    k = len(classes)    
        
    # create new list with numerical classes
    class_pts_2 = np.zeros(class_pts.shape)
    for i in range(len(classes)):
        sel = class_pts == classes[i]
        class_pts_2[sel] = i

    # reshape classification array into 2D array corresponding to grid
    class_grid = class_pts_2.reshape(len(xticks),len(yticks) )
    
    # Set a color map        
    my_cmap = plt.get_cmap(cmap)
          
    # Close any open figures and set plot size.
    
    if(close): 
        plt.close()
    if(not fig_size is None):
        plt.figure(figsize=fig_size)
    
    # Add color mesh
    plt.pcolormesh(xticks, yticks, class_grid, cmap = my_cmap, zorder = 1, 
                   vmin=0, vmax=k-1 )
    
    # Add transparency layer to lighten the colors
    plt.fill([x0,x0,x1,x1], [y0,y1,y1,y0], 'white', alpha=0.5, zorder = 2)
    
    # Select discrete cuts for color map
    cuts = np.arange(k) / (k - 1)

    # Add scatter plot for each class, with seperate colors and labels
    for i in range(k):
        sel = y == classes[i]       
        
        my_c = mplc.rgb2hex(my_cmap(cuts[i]))
        plt.scatter(X[sel,0],X[sel,1], c=my_c, edgecolor='k', 
                    zorder=3, label=classes[i])

    if(legend):
        plt.legend()
    
    if(not path is None):
        plt.savefig(path, format='png')

    if(display): 
        plt.show()

def snippet_01(fs=[12,8]):
    # Generate Data 
    sd = 1956
    np.random.seed(sd)

    N = 10
    x = np.linspace(2,8,N)
    X = x.reshape(N,1)
    y =  16 - (x - 6.5)**2 + np.random.normal(0, 1, N)
    x_curve = np.linspace(0,10,50)
    y_curve = [None]
    y_pred = [None]

    # Linear Model
    m1 = LinearRegression()
    m1.fit(x.reshape(N,1),y)
    y_curve.append(m1.predict(x_curve.reshape(50,1)))
    y_pred.append(m1.predict(x.reshape(N,1)))

    # Quadratic Model
    pt = PolynomialFeatures(2)
    Xpoly = pt.fit_transform(x.reshape(N,1))
    m2 = LinearRegression()
    m2.fit(Xpoly,y)
    y_curve.append(m2.predict(pt.transform(x_curve.reshape(50,1))))
    y_pred.append(m2.predict(Xpoly))

    # Degree 10 Model
    pt = PolynomialFeatures(10)
    Xpoly = pt.fit_transform(x.reshape(N,1))
    m3 = LinearRegression()
    m3.fit(Xpoly,y)
    y_curve.append(m3.predict(pt.transform(x_curve.reshape(50,1))))
    y_pred.append(m3.predict(Xpoly))

    # Piecewise Linear
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
    y_curve.append(pred(b_opt, x_curve))
    y_pred.append(pred(b_opt, x))
            
    # KNN Model
    m5 = KNeighborsRegressor(3)
    m5.fit(x.reshape(N,1),y)
    y_curve.append(m5.predict(x_curve.reshape(50,1)))
    y_pred.append(m5.predict(x.reshape(N,1)))

    # Plot
    titles = ['Original Data', 'Model 1: Linear Regression', 'Model 2: Quadratic Model',
              'Model 3: Degree 10 Poly', 'Model 4: PW-Linear Regression', 'Model 5: 3-Nearest Neighbors']
    
    mse = [np.mean((pred - y)**2).round(4) if pred is not None else None for pred in y_pred]
    
    x0, x1, y0, y1 = 0, 10, 0, 20
    
    plt.figure(figsize=fs)
    
    for i in range(6):
        plt.subplot(2,3,i+1)
        if i > 0:
            plt.plot(x_curve, y_curve[i], c='darkorange', zorder=1)
        plt.scatter(x, y, zorder=2)
        plt.xlim(x0,x1)
        plt.ylim(y0,y1)
        title = titles[i] + ('' if i == 0 else f'\nMSE = {mse[i]}')
        plt.title(title)
    plt.tight_layout()
    plt.show()

def snippet_02(fs=[12,8]):
    # Generate Data 
    sd = 623
    np.random.seed(sd)

    N = 150
    x1 = np.random.uniform(0,10, N)
    x2 = np.random.uniform(0,10, N)
    X = np.hstack((x1.reshape(N,1), x2.reshape(N,1)))
    z = 0.2*(x1**2 + x2**2 + 0.5*x1 + 0.5*x2 - x1*x2 - 30)
    p = 1 / (1 + np.exp(-z))
    roll = np.random.uniform(0,1,N)
    y = np.where(p < roll, 0, 1)


    top = plt.get_cmap('Oranges_r', 128)
    bottom = plt.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0.6, 1, 128)),
                        bottom(np.linspace(0, 0.75, 128))))
    cm0 = ListedColormap(newcolors, name='OrangeBlue')

    # Logistic Regression
    m1 = LogisticRegression(solver='lbfgs')
    m1.fit(X, y)

    # SVM, poly
    m2 = SVC(kernel='poly', degree=3, C=1.0, gamma='auto')
    m2.fit(X,y)

    # SVM, rbf
    m3 = SVC(kernel='rbf', gamma=0.3, C=50.0)
    m3.fit(X,y)

    # KNN
    m4 = KNeighborsClassifier(1)
    m4.fit(X,y)

    # Decision Tree
    m5 = DecisionTreeClassifier()
    m5.fit(X, y)

    # Plot
    models = [m1, m2, m3, m4, m5]
    acc = [round(m.score(X, y),3) for m in models]
    
    titles = ['Original Data', 'Model 1: Logistic Regression', 'Model 2: SVM (Poly Kernel)',
              'Model 3: SVM (RBF Kernel)', 'Model 4: 1-Nearest Neighbors', 'Model 5: Decision Tree']
    
    nticks = 200
    plt.figure(figsize=fs)
    warnings.filterwarnings("ignore")
    for i in range(6):
        plt.subplot(2,3,i+1)
        if i==0:
            plt.scatter(x1, x2, c=y, edgecolor='k', cmap=cm0)
            plt.title(f'{titles[i]}')
        else:
            plot_regions(models[i-1], X, y, num_ticks=nticks, cmap=cm0, close=False, legend=False, display=False)
            plt.title(f'{titles[i]}\nAccuracy: {acc[i-1]:.4f}')
        #plt.xlim([0,10])
        #plt.ylim([0,10])
        
    plt.tight_layout()
    plt.show()

    return 
 
def snippet_03(fs=[9,9]):
    sd = 95
    np.random.seed(sd)

    n1 = 100
    x1 = np.random.normal(3, 0.5, n1).reshape(n1,1)
    y1 = np.random.normal(6, 0.8, n1).reshape(n1,1)

    n2 = 50
    x2 = np.random.normal(5.5, 0.5, n2).reshape(n2,1)
    y2 = np.random.normal(4.5, 0.5, n2).reshape(n2,1)

    n3 = 50
    x3 = np.random.normal(6.75, 0.5, n3).reshape(n3,1)
    y3 = np.random.normal(5.75, 0.5, n3).reshape(n3,1)

    n4 = 50
    x4 = np.random.normal(6, 0.5, n4).reshape(n4,1)
    y4 = np.random.normal(3, 0.5, n4).reshape(n4,1)

    n5 = 50
    x5 = np.random.normal(8.5, 0.5, n5).reshape(n5,1)
    y5 = np.random.normal(3, 0.5, n5).reshape(n5,1)

    X1 = np.vstack((x1, x2, x3, x4, x5))
    X2 = np.vstack((y1, y2, y3, y4, y5))

    X = np.hstack((X1, X2))

    kmeans3 = KMeans(n_clusters=3)
    kmeans3.fit(X)
    y_kmeans3 = kmeans3.predict(X)

    kmeans4 = KMeans(n_clusters=4)
    kmeans4.fit(X)
    y_kmeans4 = kmeans4.predict(X)

    kmeans5 = KMeans(n_clusters=5)
    kmeans5.fit(X)
    y_kmeans5 = kmeans5.predict(X)

    plt.close()
    plt.figure(figsize=fs)

    plt.subplot(2,2,1)
    plt.scatter(X[:,0], X[:,1], s=20, cmap='rainbow', edgecolor='k')
    plt.title('Original Data')

    plt.subplot(2,2,2)
    plt.scatter(X[:,0], X[:,1],c=y_kmeans3, s=20, cmap='rainbow', edgecolor='k')
    plt.title('K-Means (K = 3)')

    plt.subplot(2,2,3)
    plt.scatter(X[:,0], X[:,1],c=y_kmeans4, s=20, cmap='rainbow', edgecolor='k')
    plt.title('K-Means (K = 4)')

    plt.subplot(2,2,4)
    plt.scatter(X[:,0], X[:,1],c=y_kmeans5, s=20, cmap='rainbow', edgecolor='k')
    plt.title('K-Means (K = 5)')

    plt.show()