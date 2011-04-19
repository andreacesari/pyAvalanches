import scipy
import scipy.ndimage as nd
import time

def getAxyLabels(labels,imageDir="Left_to_right",edgeThickness=1):
    """
    Get the edges touched by clusters 
    given by a 2D array with labels, obtained using scipy.ndimage.label
    """
    et = edgeThickness
    # Find first the 4 borders
    # Definition good for "Left_to_right"
    left = list(labels[0:et,:].flatten())
    right = list(labels[-et:,:].flatten())
    bottom = list(labels[:,0:et].flatten())
    top = list(labels[:,-et:].flatten())
    if imageDir=="Left_to_right":
        pass
    elif imageDir == "Right_to_left":
        left,right,bottom,top = right,left,top,bottom
    elif imageDir == "Bottom_to_top":
        left,right,bottom,top = bottom, top,right,left
    elif imageDir == "Top_to_bottom":
        left,right,bottom,top = top,bottom,left,right
    else:
        raise ValueError, "avalanche direction not defined"
    # Find the sets
    # Select the labels contained in the borders
    setLrbt = set(left+right+bottom+top)
    # Remove the value 0 as it is not a cluster
    setLrbt.remove(0)
    maxLabel = labels.max()
    lrbt = left,right,bottom,top
    # Prepare a list with all '0000' labels
    list_Axy = ['0000']*maxLabel
    # Search for the Axy except for the A00
    for cluster in setLrbt:
        list_Axy[cluster-1] = "".join([((cluster in edge) and '1' or '0') for edge in lrbt])
        #pixels_on_the_edge = "/".join([str(edge.count(cluster)) for edge in lrbt])
    return scipy.array(list_Axy)
        
if __name__ == "__main__":
    startTime = time.time()
    d = {}
    a0= scipy.array([],dtype='int32')
    a = scipy.loadtxt("image_test.dat",dtype='int16')
    structure = [[0, 1, 0], [1,1,1], [0,1,0]]
    #a = a.repeat(100,axis=0)
    #a = a.repeat(66,axis=1)
    labels, n = nd.label(a,structure)
    list_sizes = nd.sum(a,labels,range(1,n+1))
    array_sizes = scipy.array(list_sizes,dtype='int16')
    array_Axy = getAxyLabels(labels,edgeThickness=1)
    for Axy in set(array_Axy):
        sizes = array_sizes[array_Axy==Axy] # Not bad...
        d[Axy] = scipy.concatenate((d.get(Axy,a0),sizes))
    
    for i in zip(array_sizes,array_Axy):
        print i
    print "Done in %.3f s" % (time.time()-startTime)
    
    map