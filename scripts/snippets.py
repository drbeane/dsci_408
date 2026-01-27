import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from scipy.optimize import minimize
import warnings 

#from ipywidgets import *
from ipywidgets import FloatSlider
from IPython.display import display, HTML, Markdown

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



def plot_regions(
    model, X, y, num_ticks=100, cmap='rainbow', colors=None, alpha=1,
    fig_size=None, legend=True, display=True, path=None, keras=False):

    import matplotlib.colors as mplc
    from matplotlib.colors import LinearSegmentedColormap

    # Convert X to numpy array
    X = np.array(X)
    y = np.array(y)
    
    # Set color defaults for binary classificaiton
    if colors is None and len(np.unique(y)) == 2:
        colors=['salmon', 'cornflowerblue']


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
        prob = model.predict(grid_pts, verbose=0)
        if prob.shape[1] == 1:
            class_pts = np.where(prob < 0.5, 0, 1)
        else: 
            class_pts = np.argmax(prob, axis=1)
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
    if colors is None:        
        my_cmap = plt.get_cmap(cmap)
    else:
        my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
          
    # Close any open figures and set plot size.
    
    #if(close): 
    #    plt.close()
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
        plt.scatter(
            X[sel,0],X[sel,1], c=my_c, edgecolor='k', 
            alpha=alpha, zorder=10, label=classes[i]
        )

    plt.xlim([x0, x1])
    plt.ylim([y0, y1])

    if(legend):
        plt.legend()
    
    if(not path is None):
        plt.savefig(path, format='png')

    if(display): 
        plt.show()

def vis_training(hlist, start=1, fs=[12,4]):

    # Merge history objects    
    history = {}
    for k in hlist[0].history.keys():
        history[k] = sum([h.history[k] for h in hlist], [])

    # Determine epoch range to display
    epoch_range = range(start, len(history['loss'])+1)
    s = slice(start-1, None)

    # Determine if validation data is included
    validation = list(history.keys())[-1][:3] == 'val'

    # Determine number of plots
    n = len(history.keys()) 
    if validation: 
        n = n//2
    
    plt.figure(figsize=fs)

    for i in range(n):
        k = list(history.keys())[i]
        plt.subplot(1,n,i+1)
        plt.plot(epoch_range, history[k][s], label='Training')
        if validation:
            plt.plot(epoch_range, history['val_' + k][s], label='Validation')
        
        k = k.upper() if k in ['auc'] else k.title()
        plt.xlabel('Epoch'); plt.ylabel(k); plt.title(k + ' by Epoch')
        plt.grid()
        plt.legend()

    plt.tight_layout()
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

def data_for_456(N, sd):
    np.random.seed(sd)
    x = np.random.uniform(low=0, high=10, size=N).round(1)
    x.sort()
    y = (5 + 1.4 * x + np.random.normal(0, 2.5, N)).round(1)
    return x, y


def snippet_04(fs=[8,4]):
    x, y = data_for_456(12, 164)
    df = pd.DataFrame({'x':x, 'y':y}).T
    display(df)

    plt.figure(figsize=fs)
    plt.scatter(x, y, c='orchid', edgecolor='k', s=60, alpha=0.8)
    plt.axis((0,10,0,25))
    plt.show()

def snippet_05(fs=[6,6]):
    x, y = data_for_456(12, 164)
    
    grid_b0 = np.arange(-2.1, 12.1, 0.01)
    grid_b1 = np.arange(-1.1, 4.1, 0.01)
    [B0, B1] = np.meshgrid(grid_b0, grid_b1)
    
    yhat = B0.flatten() + x.reshape(-1,1) * B1.reshape(1,-1)
    errors = y.reshape(-1,1) - yhat
    sq_error = errors**2
    sse = sq_error.sum(axis=0)
    sse = sse.reshape(len(grid_b1),len(grid_b0))

    levels = [0, 50, 100, 500, 1000, 2000, 4000, 5000]
    plt.figure(figsize=fs)
    plt.contourf(B0, B1, sse, levels=levels, cmap='Spectral_r')
    plt.contour(B0, B1, sse, levels=levels, colors='k', linewidths=1)
    plt.title('Countour Map for SSE')
    plt.xlabel('Beta_0')
    plt.ylabel('Beta_1')
    plt.show()


def snippet_06():
    np.random.seed(164)
    N = 12
    x = np.random.uniform(low=0,high=10,size=N)
    x.sort()
    y = 5 + 1.4 * x + np.random.normal(0, 2.5, N)

    def regression_example(b, m, show_errors):
        yhat = b + m * x
        
        plt.figure(figsize=[5.5, 5.5])
        plt.plot([0,10], [b,10*m+b], c='purple')
        if show_errors:
            for i in range(len(x)):
                plt.plot([x[i],x[i]],[y[i],yhat[i]], c='black', lw=0.75,zorder=1)
        plt.scatter(x,y,zorder=2)        
        plt.axis((0,10,0,25))
        plt.show()


    def sse(b, m, **kwargs):
        yhat = b + m * x 
        errors = y - yhat
        sq_errors = errors**2
        SSE = np.sum(sq_errors)
        print('')
        sp1 = "**<span style=\"font-size:1.4em;\">"
        sp2 = "</span>**"
        display(Markdown(sp1 + "Loss Function:" + sp2))
        display(Markdown(sp1 + "SSE = " + str(round(SSE,2)) + sp2))

    def table(b, m, **kwargs):   
        yhat = b + m * x 
        errors = y - yhat
        sq_errors = errors**2
        
        df = pd.DataFrame({
            'x':x, 'y':y, 'yhat':yhat, 'error':errors, 'sq_error':sq_errors
        }).round(3)

        display(df)

    b = FloatSlider(min=-2, max=10, step=0.1, value=2, 
                continuous_update=False, layout=Layout(width='200px'))
    m = FloatSlider(min=-2, max=2, step=0.01, value=0, 
                    continuous_update=False, layout=Layout(width='200px'))
    e = Checkbox(value=False, description='Show Errors', disable=False)

    cdict = {'b':b, 'm':m, 'show_errors':e}

    plot_out = interactive_output(regression_example, cdict)
    sse_out = interactive_output(sse, cdict)
    table_out = interactive_output(table, cdict)

    controls = VBox([b, m, e, sse_out])

    display(HBox([controls, plot_out, table_out]))
    
    
def snippet_07(b, fs=[6,4]):
    exam_df = pd.DataFrame({
        'x1': [50, 55, 70, 95, 100, 120], 
        'x2': [0, 8, 18, 6, 0, 12],
        'y' : ['F', 'F', 'P', 'F', 'P', 'P']
    })
    passed = exam_df.query('y == "P"')
    failed = exam_df.query('y == "F"')

    k = -b[0] / b[2]   # intercept
    m = -b[1] / b[2]   # slope
    if b[2] > 0: 
        c_bot = 'salmon'
        c_top = 'steelblue'
    else:
        c_bot = 'steelblue'
        c_top = 'salmon'
    
    plt.figure(figsize=fs)
    plt.scatter(passed.x1, passed.x2, s=120, c='cornflowerblue', edgecolors='k', label='Passed', zorder=2)
    plt.scatter(failed.x1, failed.x2, s=120, c='salmon', edgecolors='k', label='Failed', zorder=2)
    
    
    plt.fill([0,200,200,0],[-10,-10, k + m*200, k], c_bot, alpha=0.2, zorder=1)    # below line
    plt.fill([0,200,200,0],[30,30, k + m*200, k], c_top, alpha=0.2, zorder=1)   # above line
    
    plt.plot([0,200],[k, k + 200*m], c='k', alpha=0.6, zorder=2)
    plt.xlabel('Hours Spent Studying Alone')
    plt.ylabel('Hours Spent in Seminar')
    plt.xlim([40,130])
    plt.ylim([-2,22])
    plt.show()
    
def snippet_08(title, b):
    def show_table(b):
        df = pd.DataFrame({
            'x1': [50, 55, 70, 95, 100, 120], 
            'x2': [0, 8, 18, 6, 0, 12],
            'y' : ['F', 'F', 'P', 'F', 'P', 'P'],
        })
        z = b[0] + b[1] * df.x1.values + b[2] * df.x2.values
        p = 1 / (1 + np.exp(-z))
        df['p'] = p.round(3)
        df['pi'] = np.where(df.y == 'P', p, 1 - p).round(3)
        lik = np.prod(df.pi.values)

        display(df)        
        display(Markdown(f'**Likelihood = {100*lik:.2f}%**'))
        display(Markdown(f'**NLL = {-np.log(lik):.2f}**'))

    def blank():
        display(Markdown("<html>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp</html>"))

    cdict = {}
    plot_1 = interactive_output(lambda : snippet_07(b), cdict)
    table_1 = interactive_output(lambda : show_table(b), cdict)
    blank_1 = interactive_output(blank, cdict)
    display(Markdown(f"##{title}"))
    display(HBox([plot_1, blank_1, table_1]))
 
 
def snippet_09():
    N = 6
    np.random.seed(7)
    X = np.hstack([
        np.random.uniform(0, 10, size=(N,1)).round(1),
        np.random.uniform(0, 100, size=(N,1)).round(1),
    ])
    mm_scaler = MinMaxScaler()
    Xs_mm = mm_scaler.fit_transform(X)
    plt.figure(figsize=[8,1])
    plt.scatter(X[:,0], np.ones(N), zorder=2)
    plt.scatter(X[:,0], np.zeros(N), zorder=2)
    for i in range(N): 
        plt.text(X[i,0]-0.1, 1.2, X[i,0])
        plt.text(X[i,0]-0.1, -0.4, Xs_mm[i,0].round(2))
    plt.text(-1.8, 1.1, 'Original values:')
    plt.text(-1.8, -0.3, 'Scaled values:')
    plt.vlines(X[:,0], 0, 1, color='gray', zorder=1)
    plt.axis('off')
    plt.show()
    
def snippet_10():
    N = 6
    np.random.seed(7)
    X = np.hstack([
        np.random.uniform(0, 10, size=(N,1)).round(1),
        np.random.uniform(0, 100, size=(N,1)).round(1),
    ])
    std_scaler = StandardScaler()
    Xs_std = std_scaler.fit_transform(X)
    plt.figure(figsize=[8,1])
    plt.scatter(X[:,0], np.ones(N), zorder=2)
    plt.scatter(X[:,0], np.zeros(N), zorder=2)
    for i in range(N): 
        plt.text(X[i,0]-0.1, 1.2, X[i,0])
        plt.text(X[i,0]-0.1, -0.4, Xs_std[i,0].round(2))
    plt.text(-1.8, 1.1, 'Original values:')
    plt.text(-1.8, -0.3, 'Scaled values:')
    plt.vlines(X[:,0], 0, 1, color='gray', zorder=1)
    plt.axis('off')
    plt.show()
    
def snippet_11(X, y, fs=[12,8], num_ticks=200):
    import ipywidgets as widgets
    def show_plot(max_depth, min_samples_leaf):
        temp_mod = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=1)
        temp_mod.fit(X, y)
        print('Training Accuracy:', temp_mod.score(X, y))
        plot_regions(temp_mod, X, y, fig_size=fs, num_ticks=num_ticks, colors=['salmon', 'cornflowerblue'])
        

    _ = widgets.interact(
        show_plot,
        max_depth = widgets.IntSlider(min=1,max=30,step=1,value=1,continuous_update=False),
        min_samples_leaf = widgets.IntSlider(min=1,max=30,step=1,value=1,continuous_update=False)
        
    )


def snippet_12(mod, X, y, colors, sz=300, fig_size=None, num_ticks=100, display=False, show_support=True, show_margins=True):
    plot_regions(mod, X, y, colors=colors, fig_size=fig_size, num_ticks=num_ticks, display=display)
    
    if show_support:
        plt.scatter(
            mod.support_vectors_[:, 0], mod.support_vectors_[:, 1], s=sz, 
            linewidth=1, edgecolors='k', zorder=10, facecolors='none'
        )
    xticks = np.linspace(np.min(X[:,0])-1/2, np.max(X[:,0])+1/2, 100)
    yticks = np.linspace(np.min(X[:,1])-1/2, np.max(X[:,1])+1/2, 100)
    grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
    
    if show_margins:
        P = mod.decision_function(grid_pts).reshape(100,100)
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], linestyles = ['--', '-', '--'], zorder = 4)
    if display: plt.display()


def snippet_13(X, y, colors=None, fig_size=[8,6], param_range=None, log=True, show_slider=True):
    import ipywidgets as wid

    if param_range is None:
        c0, c1, step, start = -10, 20, 1, 20
    else:
        c0, c1, step, start = param_range

    label = 'log C' if log else 'C'

    slider = wid.FloatSlider(
        min=c0, max=c1, step=step, value=start, description=label, 
        continuous_update=False, layout=wid.Layout(width='275px')
    )

    colors = ['salmon', 'cornflowerblue'] if colors is None else colors
    def svm_plot_1(slider_value):
        
        C = np.exp(slider_value) if log else slider_value
        # build model
        mod_01 = SVC(kernel='linear', C=C, )
        mod_01.fit(X, y)
        
        # plot decision regions
        
        plot_regions(mod_01, X, y, 200, colors=colors, fig_size=fig_size, display=False)
        
        # plot support vectors
        plt.scatter(mod_01.support_vectors_[:, 0], mod_01.support_vectors_[:, 1], s=300, 
                    linewidth=1, edgecolors='k', zorder=4, facecolors='none')
        
        xticks = np.linspace(np.min(X[:,0])-1/2, np.max(X[:,0])+1/2, 100)
        yticks = np.linspace(np.min(X[:,1])-1/2, np.max(X[:,1])+1/2, 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        
        P = mod_01.decision_function(grid_pts).reshape(100,100)
        
        plt.contour(xticks, yticks, P, colors='k', levels=[-1,0,1], linestyles = ['--', '-', '--'], zorder = 4)
        plt.show()

    cdict = {'slider_value':slider}
    plot_out = wid.interactive_output(svm_plot_1, cdict)

    if show_slider:
        display(wid.VBox([slider, plot_out]))
    else:
        display(plot_out)

def snippet_14(fig_size=[16,6]):
    from matplotlib.colors import ListedColormap
    import ipywidgets as wid
    
    np.random.seed(1)

    L1 = np.array([2,2])
    L2 = np.array([4,3])
    L = np.vstack([L1,L2])
    X1a, y1a = make_circles(n_samples=20, noise=0.1, factor=0.4)
    X1b, y1b = make_circles(n_samples=20, noise=0.1, factor=0.4)
    X1c, y1c = make_blobs(n_samples=10, centers=1, cluster_std=0.3)
    X1d, y1d = make_blobs(n_samples=10, centers=1, cluster_std=0.3)
    X1a = X1a + L1
    X1b = X1b + L2
    X1c = X1c + np.array([6.5,8.5])
    X1d = X1d + np.array([7.5,7.5])
    X1 = np.vstack([X1a, X1b, X1c, X1d])
    y1 = np.hstack([y1a, y1b, y1c, y1d])


    g = wid.FloatSlider(min=0.5, max=15, step=0.5, value=15, description = 'g', 
                    continuous_update=False, layout=wid.Layout(width='275px'))

    def svm_plot_1(g):

        plt.figure(figsize = fig_size)
        
        plt.subplot(1, 2, 1)
        plt.scatter(X1[:, 0], X1[:, 1],  c=y1, s=80, edgecolor='k', cmap='rainbow')
        plt.scatter(L1[0], L1[1], c='darkorange', s=120, edgecolor='k', marker='D')
        plt.scatter(L2[0], L2[1], c='darkgreen', s=120, edgecolor='k', marker='D')


        xticks = np.linspace(0, 10, 100)
        yticks = np.linspace(0, 10, 100)
        grid_pts = np.transpose([np.tile(xticks,100), np.repeat(yticks,100)])
        D1 = np.exp(-g*(np.sum((grid_pts - L1)**2, axis=1)**0.5).reshape(100,100))
        D2 = np.exp(-g*(np.sum((grid_pts - L2)**2, axis=1)**0.5).reshape(100,100))
        
        base = plt.get_cmap('Oranges')
        subset = base(np.linspace(0.45, 0.75))
        cm0 = ListedColormap(subset, name='Custom')
        
        base = plt.get_cmap('Greens')
        subset = base(np.linspace(0.45, 0.75))
        cm1 = ListedColormap(subset, name='Custom')
        
        
        plt.contour(xticks, yticks, D1, cmap=cm0, levels=[0.001, 0.01, 0.1], linestyles = ['--', '--', '--'], zorder = 4)
        plt.contour(xticks, yticks, D2, cmap=cm1, levels=[0.001, 0.01, 0.1], linestyles = ['--', '--', '--'], zorder = 4)
        plt.xlim(0,6)
        plt.ylim(0,4.5)
        plt.title('Original Features')

        T1 = np.exp(-g*np.sum((X1 - L1)**2, axis=1)**0.5).reshape(-1,1)
        T2 = np.exp(-g*np.sum((X1 - L2)**2, axis=1)**0.5).reshape(-1,1)
        T = np.hstack([T1, T2])
        
        plt.subplot(1, 2, 2)
        plt.scatter(T[:, 0], T[:, 1],  c=y1, s=80, edgecolor='k', cmap='rainbow')
        plt.title('Transformed Features')
        plt.show()

    cdict = {'g':g}
    plot_out = wid.interactive_output(svm_plot_1, cdict)

    display(wid.VBox([g, plot_out]))
    
def snippet_15(fs=[15,5]):

    import numpy as np
    from sklearn.decomposition import PCA

    np.random.seed(1)
    n = 50
    v0 = np.random.normal(0, 1.2, [n,1])
    v1 = np.random.normal(0, 0.6, [n,1])
    X = np.hstack([v0 + v1, 0.8*v0 - v1])

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    pc = pca.components_

    mu = np.mean(X, axis=0)

    a = mu + pc[0, :]
    b = mu + pc[1, :]

    c0 = 'darkgray'
    c1 = 'tab:blue'
    c2 = 'darkorange'
    c3 = 'gold'
    c4 = 'forestgreen'
    c5 = 'firebrick'

    plt.figure(figsize=fs)

    ###################################
    # Figure 1
    ###################################
    plt.subplot(1, 3, 1)
    plt.scatter(X[:,0], X[:,1], c=c0)
    plt.arrow(mu[0], mu[1], pc[0,0], pc[0,1], width=0.12, length_includes_head=True, facecolor=c1)
    plt.arrow(mu[0], mu[1], pc[1,0], pc[1,1], width=0.12, length_includes_head=True, facecolor=c2)
    plt.scatter(mu[0], mu[1], c=c3, edgecolor='k', s=90, marker='o')
    plt.gca().set_aspect('equal')
    plt.xlim([-4,4]); plt.ylim([-4,4])
    plt.xlabel('X1'); plt.ylabel('X2'); 
    plt.title('Principal Components')

    ###################################
    # Figure 2
    ###################################
    n1 = 19
    n2 = 11
    p1 = mu + Z[n1, 0] * pc[0, :] 
    p2 = mu + Z[n2, 0] * pc[0, :]
    q1 = p1 + Z[n1, 1] * pc[1, :] 
    q2 = p2 + Z[n2, 1] * pc[1, :]

    plt.subplot(1, 3, 2)
    plt.scatter(X[:,0], X[:,1], c=c0)
    plt.arrow(mu[0], mu[1], pc[0,0], pc[0,1], width=0.12, length_includes_head=True, facecolor=c1, zorder=8)
    plt.arrow(mu[0], mu[1], pc[1,0], pc[1,1], width=0.12, length_includes_head=True, facecolor=c2, zorder=8)

    plt.plot([mu[0], p1[0]], [mu[1], p1[1]], linewidth=2, color=c1, linestyle='-')
    plt.plot([mu[0], p2[0]], [mu[1], p2[1]], linewidth=2, color=c1, linestyle='-')
    plt.plot([p1[0], q1[0]], [p1[1], q1[1]], linewidth=2, color=c2, linestyle='-')
    plt.plot([p2[0], q2[0]], [p2[1], q2[1]], linewidth=2, color=c2, linestyle='-')

    plt.scatter(mu[0], mu[1], c=c3, edgecolor='k', s=90, zorder=10)
    plt.scatter(X[n1, 0], X[n1, 1], c=c4, edgecolor='k', s=90, zorder=10)
    plt.scatter(X[n2, 0], X[n2, 1], c=c5, edgecolor='k', s=90, zorder=10)
    plt.gca().set_aspect('equal')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('X1'); plt.ylabel('X2'); 
    plt.title('Principal Component Decomposition')

    ###################################
    # Figure 3
    ###################################
    plt.subplot(1, 3, 3)
    plt.scatter(Z[:,0], Z[:,1], c=c0)

    plt.arrow(0, 0, 1, 0, width=0.12, length_includes_head=True, facecolor=c1, zorder=8)
    plt.arrow(0, 0, 0, 1, width=0.12, length_includes_head=True, facecolor=c2, zorder=8)

    plt.plot([0, Z[n1, 0]], [0, 0], linewidth=2, color=c1, linestyle='-')
    plt.plot([0, Z[n2, 0]], [0, 0], linewidth=2, color=c1, linestyle='-')

    plt.plot([Z[n1, 0], Z[n1, 0]], [0, Z[n1, 1]], linewidth=2, color=c2, linestyle='-')
    plt.plot([Z[n2, 0], Z[n2, 0]], [0, Z[n2, 1]], linewidth=2, color=c2, linestyle='-')


    plt.scatter(0, 0, c=c3, edgecolor='k', s=90, zorder=10)
    plt.scatter(Z[n1, 0], Z[n1, 1], c=c4, edgecolor='k', s=90, zorder=10)
    plt.scatter(Z[n2, 0], Z[n2, 1], c=c5, edgecolor='k', s=90, zorder=10)
    plt.gca().set_aspect('equal')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('Z1'); plt.ylabel('Z2'); 
    plt.title('Transformed Coordinates')

    plt.show()
    
def snippet_16(digit, pca, Z, n_frames=150):
    from IPython.display import Image
    import imageio
    import os
    import shutil

    if 'frames' in os.listdir():
        shutil.rmtree('frames/')
    os.mkdir('frames/')
    img = np.zeros(shape=(28, 28))
    for i in range(n_frames):
        pc = pca.components_[i,:]
        step = Z[digit, i] * pc.reshape(28,28)
        img += step
        plt.figure(figsize=[4,4])
        plt.imshow(img, cmap='Greys')
        plt.axis('off')
        plt.savefig(f'frames/frame_{i}.png', transparent=False, facecolor='white')
        plt.close()

    frames = []
    for i in range(n_frames):
        image = imageio.v2.imread(f'frames/frame_{i}.png')
        frames.append(image)

    imageio.mimsave('./example.gif', frames, fps = 20)       

    display(Image(data=open('example.gif','rb').read(), format='png'))
    
def snippet_17(fs=[9,3]):
    x = np.linspace(-10,10,100)
    y1 = 1 / (1 + np.exp(-x))
    y2 = np.where(x<0, 0, x)

    plt.figure(figsize=fs)
    plt.subplot(1,2,1)
    plt.plot([-10,10],[1,1], linestyle=':', c="r")
    plt.plot([-10,10],[0,0], linestyle=':', c="r")
    plt.plot([0,0],[0,1], linewidth=1, c="dimgray")
    plt.title('Sigmoid Activation Function')
    plt.plot(x, y1, linewidth=2)

    plt.subplot(1,2,2)
    plt.plot([0,0],[0,10], linewidth=1, c="dimgray")
    plt.plot([-10,10],[0,0], linewidth=1, c="dimgray")
    plt.plot(x, y2, linewidth=2)
    plt.title('ReLu Activation Function')
    plt.show()

if __name__ == "__main__":

    snippet_08('Model 1', b=[-2.4, 0.016, 0.1])
