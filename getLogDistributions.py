from pylab import load,  save,  loglog,  show
import scipy
import scipy.stats
from scipy import array
import numpy as np
import itertools

def checkIfVoid(listValues):
    # Check the list of Values
    if not len(listValues):
        raise "Please pass a list of values"
    else:
        return True
    
def getLogBins(first_point, last_point, log_step):
    """
    get the bin in log scale and the center bin value
    
    Parameters:
    ----------------
    first_point, last_point : number
    First and last point of the x-axis
    
    log_step : number
    Required log-distance between x-points
    
    Returns:
    -----------
    xbins : array of the x values at the center (in log-scale) of the bin
    bins : array of the x values of the bins 
    """
    log_first_point = scipy.log10(first_point)
    log_last_point = scipy.log10(last_point)
    # Calculate the bins as required by the histogram function, i.e. the bins edges including the rightmost one
    N_log_steps = scipy.floor((log_last_point-log_first_point)/log_step) + 1.
    llp = N_log_steps * log_step + log_first_point
    bins_in_log_scale = np.linspace(log_first_point, llp, N_log_steps+1)
    bins = 10**bins_in_log_scale
    center_of_bins_log_scale = bins_in_log_scale[:-1] + log_step/2.
    xbins = 10**center_of_bins_log_scale
    return xbins, bins

def logDistribution(listValues, log_step=0.2, first_point=None, last_point=None, normed=True):
    """
    Calculate the distribution in log scale from a list of values
    """
    # Check the list of Values
    if not checkIfVoid(listValues):
        print("Error")
    if not first_point:
        first_point = scipy.amin(listValues)
    if not last_point:
        last_point = scipy.amax(listValues)
    xbins, bins = getLogBins(first_point, last_point, log_step)
    yhist = scipy.stats.stats.histogram2(listValues, bins)
    deltas = bins[1:]-bins[:-1]
    yhist = yhist[:-1]/deltas
    if normed:
        yhist = yhist/scipy.sum(yhist)
    return xbins, yhist
    
def averageLogDistribution(values, log_step=0.2, first_point=None, last_point=None):
    """
    calculates the <values> vs. xVariable in log scale

    Parameters:
    ---------------
    values : dict
    A dictionary where the keys are the xValues
    and each element contains an array-like sequence of data
    Example:
    [1: array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
     2: array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2]),
     
    values : ndarray
    Two columns array of xValues and yValues,
    to be rearranged as above
     
    Returns:
    center point of the bin, average value within the bin

    """
    # Check the list of Values
    if not checkIfVoid(values):
        print("Error")
    if isinstance(values, dict):
        xValues = np.asarray(values.keys())
        yValues = np.asarray(values.values())
    elif isinstance(values, np.ndarray):
        xValues = np.unique(values[:, 0])
        yValues = []
        for xVal in xValues:
            index = values[:,0] == xVal
            yValues.append(values[index, 1])
        yValues = scipy.array(yValues)
    else:
        print("Values shape not recognized")
        return
    if not first_point:
        first_point = scipy.amin(xValues)*0.99
    if not last_point:
        last_point = scipy.amax(xValues)*1.01

    xbins, bins = getLogBins(first_point, last_point, log_step)
    yAverage = []
    for i, j in zip(bins[:-1],bins[1:]):
        q1, q2 = np.greater_equal(xValues, i), np.less(xValues, j)
        q = np.logical_and(q1, q2)
        if sum(q) == 0:
            averageValue = np.NaN
        else:
            allElements = [val for val in itertools.chain(*yValues[q])]
            averageValue = sum(allElements)/float(len(allElements))
            #print averageValue, allElements
        yAverage.append(averageValue)
    yAverage =  np.asanyarray(yAverage)
    # Check if there are NaN values
    iNan = np.isnan(yAverage)
    x = xbins[~iNan]
    y = yAverage[~iNan]
    return x, y
     
if __name__ == "__main__":
    #listValues = scipy.rand(100)*230
    #listValues = scipy.loadtxt("/home/gf/Python/Moke/wtm_sizes.dat")
    #xbins, yhist = logDistribution(listValues)
    #loglog(xbins,yhist,'o')
    #show()
     
    
    #x,y = averageLogDistribution(N_cluster)
    #loglog(x,y, 'bo')
    #show()
    
    q = np.array([[    1,     1],
       [    1,     1],
       [    2,     2],
       [    5,     1],
       [    1,     1],
       [    6,     3],
       [32258,    84],
       [22689,   458],
       [  520,    25],
       [  510,    16],
       [ 1215,     6],
       [   20,    10],
       [   38,    12],
       [ 1324,     9],
       [  104,    14],
       [   44,    12],
       [  512,     8],
       [   56,    15],
       [ 4989,     3],
       [   17,     9],
       [   11,     5],
       [   19,     6],
       [  271,     1],
       [    7,     2],
       [    4,     2],
       [   77,     3],
       [    1,     1],
       [  389,     4],
       [   49,    11],
       [ 2921,     5],
       [    7,     4],
       [    8,     3],
       [    2,     1],
       [  285,     3],
       [    1,     1],
       [    1,     1],
       [   55,     2],
       [    8,     6],
       [ 2346,     5],
       [  127,    27],
       [  201,    13],
       [  742,    10],
       [   22,     8],
       [   14,     7],
       [ 1439,     6],
       [   17,     8],
       [    3,     3],
       [   28,    12],
       [ 1167,     2],
       [   67,     9],
       [   47,     5],
       [    1,     1],
       [    1,     1],
       [   92,     1],
       [    3,     1],
       [    7,     1],
       [    7,     3],
       [   19,     2],
       [    2,     1],
       [   97,     1],
       [    6,     1],
       [    3,     1],
       [    1,     1],
       [   47,     7],
       [  354,     5],
       [    3,     2],
       [    2,     1],
       [    3,     1],
       [ 1182,     1],
       [    1,     1],
       [    1,     1],
       [   12,     1],
       [    2,     1],
       [   17,     3],
       [    4,     1],
       [   12,     1],
       [    1,     1],
       [   83,     7],
       [    3,     1],
       [  156,     2],
       [    1,     1],
       [  167,     2],
       [  114,     3],
       [    2,     1],
       [  105,     2],
       [    3,     1],
       [   79,     4],
       [  162,    16],
       [  876,     7],
       [   27,     6],
       [    6,     3],
       [  275,     5],
       [   75,     8],
       [    4,     1],
       [    1,     1],
       [   74,     1],
       [   10,     3],
       [   11,     4],
       [ 1019,     1],
       [   13,     6],
       [   12,     4],
       [    6,     2],
       [    6,     4],
       [  483,     5],
       [  377,     3],
       [ 1346,     6],
       [ 1071,     5],
       [  163,     6],
       [    3,     2],
       [  143,     2],
       [   24,     6],
       [ 1500,    15],
       [  712,    39],
       [   28,     8],
       [    5,     3],
       [    4,     3],
       [   63,     6],
       [  726,     1],
       [   71,     1],
       [ 5492,     5],
       [   20,     3],
       [    3,     1],
       [    3,     1],
       [   19,     2],
       [    3,     1],
       [    1,     1],
       [    1,     1],
       [  125,     1],
       [    3,     1],
       [    7,     3],
       [  247,     1],
       [    3,     1],
       [    2,     1],
       [   11,     1],
       [   15,     3],
       [    3,     1],
       [    2,     2],
       [    7,     3],
       [  279,     6],
       [  131,     6],
       [    5,     1],
       [ 1565,     2],
       [   12,     1],
       [  296,     1],
       [   21,     3],
       [  118,     3],
       [   13,     1],
       [   53,     3],
       [   57,    11],
       [   75,     8],
       [    2,     1],
       [   89,     1],
       [    1,     1],
       [   57,     6],
       [   96,     4],
       [   23,    10],
       [  181,     3],
       [    2,     1],
       [    1,     1],
       [   10,     1],
       [    2,     1],
       [   26,     4],
       [    4,     2],
       [   41,     1],
       [  463,     4],
       [    1,     1],
       [  134,     1],
       [   15,     6],
       [ 4452,     6],
       [  564,    35],
       [ 2090,    18],
       [13804,    10],
       [ 1050,     4],
       [   71,     4],
       [  964,    11],
       [ 7471,     5],
       [  216,     6],
       [  103,    13],
       [   41,    13],
       [  793,     8],
       [  764,     6],
       [  463,     6],
       [   10,     5],
       [  467,    12],
       [  904,     6],
       [   77,     7],
       [    3,     1],
       [  169,    11],
       [  234,     8],
       [   16,     2],
       [    4,     2],
       [ 1432,     3],
       [    1,     1],
       [  141,     2],
       [   64,    17],
       [  203,     4],
       [  102,     7],
       [   51,     7],
       [    2,     1],
       [    2,     1],
       [ 3445,     2],
       [   52,     2],
       [    2,     1],
       [  180,     1],
       [    6,     1],
       [ 1017,     2],
       [    1,     1],
       [  904,     4],
       [ 4740,     4],
       [  482,    24],
       [  165,     1],
       [    1,     1],
       [  228,     1],
       [    7,     1],
       [   11,     3],
       [  807,     2],
       [    1,     1],
       [    1,     1],
       [   34,     2],
       [   41,     1],
       [    1,     1],
       [    2,     1],
       [    2,     1],
       [    2,     1],
       [   16,     5],
       [   30,     2],
       [    1,     1],
       [    2,     2],
       [37088,     3],
       [   80,    16],
       [   17,     7],
       [   35,     9],
       [  235,     3],
       [   82,     3],
       [    2,     1],
       [  660,     8],
       [  131,    17],
       [    4,     1],
       [  303,     3],
       [   16,     6],
       [  340,     5],
       [  201,    13],
       [25939,    14],
       [17963,     7],
       [    3,     3],
       [    7,     4],
       [    8,     4],
       [  159,     7],
       [ 1810,     2],
       [    4,     1],
       [    1,     1],
       [   57,     3],
       [  954,     2],
       [   18,     4],
       [    6,     5],
       [  100,     2],
       [    1,     1],
       [    2,     1],
       [    1,     1],
       [  287,     1],
       [  142,     1],
       [    8,     3],
       [    9,     6],
       [  124,     5],
       [    2,     1],
       [  629,     2],
       [    3,     1],
       [    4,     1],
       [    2,     1],
       [    9,     1],
       [    2,     1],
       [    2,     1],
       [ 1378,     2],
       [ 9836,     1],
       [   29,     5],
       [   38,     1],
       [    6,     2],
       [  170,     1],
       [   38,     4],
       [   21,     7],
       [    7,     4],
       [    2,     2],
       [ 5825,     1],
       [    2,     1],
       [    5,     1],
       [    1,     1],
       [    3,     1],
       [    1,     1],
       [    2,     1]])

    
    
    import random
    import matplotlib.pylab as plt
    imageSize = 500**2
    N = 100
    sizes = np.random.uniform(1, imageSize, N)
    nCluster = scipy.array([int(s**0.5)*(1+random.randint(-5,5)/100.) for s in sizes])
    #q = np.array(zip(sizes, nCluster))
    x,y = averageLogDistribution(q)
    plt.loglog(x,y, 'bo')
    plt.grid()
    plt.show()
    