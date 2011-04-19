from pylab import load,  save,  loglog,  show
import scipy

def logDistribution(listValues,log_step=0.2,first_point=1.,last_point=None,normed=True):
    """
    Calculate the distribution in log scale from a list of values
    """
    # Check the list of Values
    if not len(listValues):
        raise "Please pass a list of values" 
    if not first_point:
        first_point = scipy.amin(listvalues)
    if not last_point:
        last_point = scipy.amax(listValues)
    log_first_point = scipy.log10(first_point)
    log_last_point = scipy.log10(last_point)
    # Calculate the bins as required by the histogram function, i.e. the bins edges including the rightmost one
    N_log_steps = scipy.floor((log_last_point-log_first_point)/log_step)+1.
    bins_in_log_scale = scipy.arange(log_first_point,log_step*(N_log_steps+1),log_step)
    center_of_bins_log_scale = bins_in_log_scale[:-1]+log_step/2.
    bins = 10**bins_in_log_scale
    xbins = 10**center_of_bins_log_scale
    yhist = scipy.stats.stats.histogram2(listValues,bins)
    deltas = bins[1:]-bins[:-1]
    yhist = yhist[:-1]/deltas
    if normed:
        yhist = yhist/scipy.sum(yhist)
    return xbins, yhist
    
if __name__ == "__main__":
    #listValues = scipy.rand(100)*230
    listValues = scipy.loadtxt("wtm_sizes.dat")
    xbins, yhist = logDistribution(listValues)
    loglog(xbins,yhist,'o')
    show()