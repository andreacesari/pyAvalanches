import scipy
import scipy.ndimage as nd
import scipy.stats.stats
import numpy as np
import numpy.ma as ma
import Image, ImageDraw
import os, sys, glob
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tables
import time
import getLogDistributions as gLD
reload(gLD)
import getAxyLabels as gAL
reload(gAL)

class StackImages:
    """
    Load and analyze a sequence of images 
    as a multi-dimensional scipy 3D array.
    The k-th element of the array (i.e. myArray[k])
    is the k-th image of the sequence.
    """
        
    def __init__(self,mainDir,filtering="Gauss",sigma=2,resize_factor=None,\
                 mountDir="/home/gf/meas/",fileType=None,imageFirst=0,imageLast=-1):
        self.colorImage = None
        self.colorImageDone = False
        self.threshold = 0
        if imageLast == None:
            imageLast = -1
        # Make a kernel as a step-function
        self.kernel = np.array([-1]*(5) +[1]*(5)) # Good for Black_to_White change of grey scale
        self.kernel0 = np.array([-1]*(5) +[0] + [1]*(5)) 
        if not fileType:
            # Check if ~/meas is mounted or existing
            who = os.getlogin()
            if who == 'gf':
                mountDir = os.path.join("/home", who, "meas")
                if not os.path.ismount(mountDir) and not os.path.isdir(mainDir):
                    print "Please mount dir ", mountDir
                    sys.exit()
            # Collect the list of images in mainDir
            extSeq = ['tif','jpg','jpeg','ppm']
            for ext in extSeq:
                dirImages = os.path.join(mainDir,"*."+ext)
                imageFileNames = sorted(glob.glob(dirImages))
                if len(imageFileNames):
                    print "Found %d images to load in %s" % (len(imageFileNames), mainDir)
                    break
            if not len(imageFileNames):    
                print "Warning, no images in %s" % mainDir
                sys.exit()
    
            # Load the images
            print "Loading images: "
            seqImages = []
            for k, image in enumerate(imageFileNames[imageFirst:imageLast]):
                im = Image.open(image)
                if resize_factor:
                    imX, imY = im.size
                    im = im.resize((imX/resize_factor, imY/resize_factor), Image.ANTIALIAS)
                # Put the image in a scipy.array of 8/16 bits
                imArray = scipy.array(im,dtype="int16")
                if filtering == "Gauss":
                    imArray = nd.gaussian_filter(imArray,sigma)
                elif filtering == "FourierGauss":
                    imArray = nd.fourier_gaussian(imArray,sigma)
                seqImages.append(imArray)        
            # Make the sequence of the images as a 3D scipy.array          
            print "Stacking ..."
            self.Array = scipy.array(tuple(seqImages))
            # Check for the grey direction
            grey_first_image = scipy.mean(self.Array[0].flatten())
            grey_last_image = scipy.mean(self.Array[-1].flatten())
            print "grey scale: %i, %i" % (grey_first_image, grey_last_image)
            if grey_first_image > grey_last_image:
                self.kernel = -self.kernel
                self.kernel0 = -self.kernel0
        elif fileType=="hdf5":
            self.hdf5 = tables.openFile("/home/gf/meas/Barkh/Film_CoFe.h5",'a')
            self.imagesObj = self.hdf5.root.tk_20nm.mg_20x.run_15.dir_down.im_Gauss25
            self.Array = self.imagesObj.read()
            if self.imagesObj._v_attrs.GREY_DIRECTION == "White_to_black":
                self.kernel = [-k for k in self.kernel]
            self.imageDir = self.imagesObj._v_attrs.IMAGE_DIRECTION
            #hdf5.close()
        self.n_images, self.dimX, self.dimY = self.Array.shape
        print "%i image(s) loaded, of %i x %i pixels" % (self.n_images, self.dimX, self.dimY)
        
    def __get__(self):
        return self.Array
        
    def __getitem__(self,i):
        "Get the i-th image"
        return self.Array[i]
    
    def imShow(self,i):
        "Show the i-th image with plt"
        plt.imshow(self.Array[i],plt.cm.gray)
        plt.draw()
        plt.show()
        
    def pixelTimeSequence(self,pixel=(0,0)):
        "Get the temporal sequence of the gray level of a pixel"
        x,y = pixel
        return self.Array[:,x,y]
        
    def pixelTimeSequenceShow(self,pixel=(0,0),width='all',useKernel='both'):
        """
        Plot the temporal sequence of the gray level of a pixel;
        width indicates the number of points before and after the switch
        width = small (half of the step width) | all (the point in the sequence)
        """
        # Plot the temporal sequence first
        pxt = self.pixelTimeSequence(pixel)
        plt.plot(pxt,'-o')
        # Add the kernel (step) function
        useKernels = ['step','zero']
        kernels = [self.kernel, self.kernel0]
        for k,kernel in enumerate(useKernels):	
            switch, (value_left, value_rigth) = self.getSwitchTime(pixel,width,useKernel=kernel)
            print "switch",kernel,"Kernel = ", switch
            print "gray level change at switch = ", abs(value_left-value_rigth)
            if width == 'small':
                halfWidth = len(kernels[k])/2
                x = range(switch-halfWidth+1, switch+halfWidth+1)
                y = np.concatenate((left*abs(kernels[k][:halfWidth]), rigth*abs(kernels[k][halfWidth:])))
            elif width=='all':
                x = range(len(pxt))
                y = (switch+1*(k==0))*[value_left]+[(value_left+value_rigth)/2.]*(k==1)+(len(pxt)-switch-1)*[value_rigth]
            plt.plot(x,y)
        plt.draw()
        plt.show()
        
    def getSwitchTime(self,pixel=(0,0),width='all',method="convolve1d",useKernel='both'):
        """
        This method searches a step in a function:
        return its position and the lower/upper values (as a tuple)
        * width: how many points are taken to calculate the levels: 
        small: len(self.kernel/2)
        all: all the data from pxTimeSeq
        Kernel choices:
        step = [1]*5 +[-1]*5
        zero = [1]*5 +[0] + [-1]*5
        both
        """
        startTime = time.time()
        pxTimeSeq = self.pixelTimeSequence(pixel)
        if method == "convolve1d":
            if useKernel == 'step' or useKernel == 'both':
                convolution_of_stepKernel = nd.convolve1d(pxTimeSeq,self.kernel)
                minStepKernel = convolution_of_stepKernel.min()
                switchStepKernel = convolution_of_stepKernel.argmin()
                switch = switchStepKernel
            if useKernel == 'zero' or useKernel == 'both':
                convolution_of_zeroKernel = nd.convolve1d(pxTimeSeq,self.kernel0)
                minZeroKernel = convolution_of_zeroKernel.min()
                switchZeroKernel = convolution_of_zeroKernel.argmin()
                switch = switchZeroKernel
            if useKernel == 'both':
                if minStepKernel <= minZeroKernel:
                    switch = switchStepKernel
                else:
                    switch = switchZeroKernel
                    leftLevel = np.int(np.mean(pxTimeSeq[0:switch])+0.5)
                    rightLevel = np.int(np.mean(pxTimeSeq[switch+1:])+0.5)
                    middle = (leftLevel+rightLevel)/2
                    rightLevelStep = np.int(np.mean(pxTimeSeq[switchStepKernel+1:])+0.5)
                    if abs(pxTimeSeq[switch]-middle)>abs(pxTimeSeq[switch]-rightLevelStep):
                        switch = switchStepKernel                    
                    #switch = (switch-1)*(pxTimeSeq[switch]<middle)+switch*(pxTimeSeq[switch]>=middle)
                #switch = switchStepKernel * (minStepKernel<=minZeroKernel/1.1) + switchZeroKernel * (minStepKernel >minZeroKernel/1.1)
        else:
            raise RuntimeError("Method not yet implemented")            
        # Get points before the switch
        if width == 'small': # TODO: fix this in case of both kernels
            halfWidth = len(self.kernel)/2
            lowPoint = switch-halfWidth+1
            if lowPoint <0:
                lowPoint = 0
            highPoint = switch+halfWidth+1
            if highPoint > len(pxTimeSeq):
                highPoint = len(pxTimeSeq)
        elif width == 'all':
            lowPoint, highPoint = 0, len(pxTimeSeq)
        else:
            print 'Method not implement yet'
            return None
        leftLevel = np.int(np.mean(pxTimeSeq[lowPoint:switch+1])+0.5)
        rigthLevel = np.int(np.mean(pxTimeSeq[switch+1:highPoint])+0.5)
        return switch, (leftLevel, rigthLevel)

    def imDiff(self,i,j=0):
        "Properly rescaled difference between images"
        im = self.Array[i]-self.Array[j]
        imMin = scipy.amin(im)
        imMax = scipy.amax(im)
        im = scipy.absolute(im-imMin)/float(imMax-imMin)*255
        return scipy.array(im,dtype='int16')

    def imDiffShow(self,i,j):
        "Show a properly rescale difference between images"
        plt.imshow(self.imDiff(i,j),plt.cm.gray)
        plt.show()

    def imDiffSave(self,mainDir):
        dirSeq = os.path.join(mainDir,"Seq")
        if not os.path.isdir(dirSeq):
            os.mkdir(dirSeq)
        n = self.n_images
        for i in range(n-1):
            im = self.imDiff(i+1,i)
            imPIL = scipy.misc.toimage(im)
            fileName = "imDiff_%i_%i.tif" % (i+1,i)
            imageFileName = os.path.join(dirSeq, fileName)
            imPIL.save(imageFileName)
        

    def contrastStretching(self,imageNum,val_1,val_2,relative=False):
        """
        Apply contrast Stretching
        to a single image
        as suggested on DigitalImageProcessing, page. 85
        """
        im = self.Array[imageNum]
        imOut = 0
        if relative:
            k = 255
        else:
            k = 1
        r1,s1 = int(val_1[0]*k), int(val_1[1]*k)
        r2,s2 = int(val_2[0]*k), int(val_2[1]*k)
        lt = scipy.less(im,r1+1)
        if r1 != 0:
            imOut += lt*im*s1/r1
        bw = scipy.greater_equal(im,r1) & scipy.less_equal(im,r2)
        if r2!=r1:
            imOut += bw*((im-r1)/float(r2-r1)*(s2-s1)+s1)
        gt = scipy.greater(im,r2)
        if r2!= 255:
            imOut += gt*((im-r2)/(255.-r2)*(255.-s2)+s2)  
        return imOut

    def histogramEqualization(self,imageNum):
        """
        Perform the histogram equalization on the image;
        returns an array
        """
        im = self.Array[imageNum]
        histOut = scipy.histogram(im.flat, range(257),normed=True)
        cdf = scipy.cumsum(histOut[0])*255
        return scipy.array(cdf[im], dtype='int16')
    
    def histogramEqualizationSequence(self):
        """
        Perform the histogram equalization on all images
        of a sequence; returns a 3D array
        """
        seqImages = []
        for i in range(self.n_images):
            im = self.Array[i]
            imOut = histogramEqualization(im)
            seqImages.append(imOut)
        return scipy.array(tuple(seqImages))
            
    def shape(self):
        seq = self.Array
        return seq.shape
    
    def getColorImage(self, useKernel='both',width='all'):
        """
        Calculate the switch times and the gray level changes
        for each pixel in the image sequence
        """
        self.switchTimes = []
        self.switchSteps = []
        noSwitch = False
        startTime = time.time()
        # ####################
        # TODO: make here a parallel calculus
        for x in range(self.dimX):
            # Print current row
            if not (x+1)%10:
                strOut = 'Analysing row:  %i/%i on %f seconds\r' % (x+1, self.dimX, time.time()-startTime)
                sys.stdout.write(strOut)
                sys.stdout.flush()
                startTime = time.time()
            for y in range(self.dimY):
                switch, levels = self.getSwitchTime((x,y), width, useKernel=useKernel)
                grayChange = np.abs(levels[0]- levels[1])
                if switch == 0: # TODO: how to deal with steps at zero time
                    print x,y
                self.switchTimes.append(switch)
                self.switchSteps.append(grayChange)
        print "\n"
        # Calculate the colours, considering the range of the switch values obtained 
        self.min_switch = np.min(self.switchTimes)
        self.max_switch = np.max(self.switchTimes)
        print "Avalanches occur between frame %i and %i" % (self.min_switch, self.max_switch)
        nImagesWithSwitch = self.max_switch - self.min_switch+1
        print "Gray changes are between %s and %s" % (min(self.switchSteps), max(self.switchSteps))
        # Prepare the Korean Palette
        self.koreanPalette = np.array([self.getKoreanColors(i-self.min_switch, nImagesWithSwitch) for i in range(self.min_switch,self.max_switch+1)])
        self.colorImageDone = True
        return

    def getKoreanColors(self,switchTime,n_images=None):
        """
        Make a palette in the korean style
        """
        if not n_images:
            n_images = self.n_images
        n = float(switchTime)/float(n_images)*3.
        R = (n<=1.)+ (2.-n)*(n>1.)*(n<=2.)
        G = n*(n<=1.)+ (n>1.)*(n<=2.)+(3.-n)*(n>2.)
        B = (n-1.)*(n>=1.)*(n<2.)+(n>=2.)
        R, G, B = [int(i*255) for i in [R,G,B]]
        return R,G,B
    
    def checkColorImageDone(self,ask=True):
        print "You must first run the getColorImage script: I'll do that for you"
        if ask:
            yes_no = raw_input("Do you want me to run the script for you (y/N)?")
            yes_no = yes_no.upper()
            if yes_no != "Y":
                return
        self.getColorImage()
        return

    def showColorImage(self,threshold=None, palette='korean',noSwitchColor='black',ask=False):
        """
        showColorImage([threshold, palette, noSwitchColor, ask])
        Show the calculated color Image of the avalanches.
        Run getColorImage if not done before.
        
        Parameters
        ---------------
        threshold: integer, optional
            Defines if the pixel switches when gray_level_change >= threshold
        palette: string, required, default = 'korean'
            Choose a palette between 'korean', 'randomKorean', and 'random'
            'randomKorean' is a random permutation of the korean palette
            'random' is calculated on the fly, so each call of the method gives different colors
        noSwithColor: string, optional, default = 'black'
            background color for pixels having gra_level_change below the threshold
            
        """
        colorPixelsArray = []
        if not threshold:
            threshold = 0
        if not self.colorImageDone:
            self.checkColorImageDone(ask=False)
        if palette == 'korean':
            pColor = self.koreanPalette
        elif palette == 'randomKorean':
            pColor = np.random.permutation(self.koreanPalette)
        elif palette == 'random':
            pColor = np.random.randint(0,256, self.koreanPalette.shape)
        if noSwitchColor == 'black':
            noSwitchColorValue = [0,0,0]
        elif noSwitchColor == 'white':
            noSwitchColorValue = [255, 255, 255]
        pColor = np.concatenate((pColor, [noSwitchColorValue]))
        # Use masked arrays to fill the background color
        isPixelSwitched = scipy.array(self.switchSteps) >= threshold
        maskedSwitchTimes = ma.array(self.switchTimes, mask = ~isPixelSwitched)
        # Move to the first switch time
        maskedSwitchTimes = maskedSwitchTimes - self.min_switch
        # Set the non-switched pixels to use the last value of the pColor array, i.e. noSwitchColorValue
        out = maskedSwitchTimes.filled(-1) # Isn't it fantastic?
        # Get the color from the palette and reshape to get the image
        self.colorImage= pColor[out].reshape(self.dimX, self.dimY, 3)
        imOut = scipy.misc.toimage(self.colorImage)
        imOut.show()
        # Count the number of the switched pixels
        switchPixels = np.sum(isPixelSwitched)
        totNumPixels = self.dimX*self.dimY
        noSwitchPixels = totNumPixels - switchPixels
        swPrint = (switchPixels, switchPixels/float(totNumPixels)*100., noSwitchPixels, noSwitchPixels/float(totNumPixels)*100.)
        print "There are %d (%.2f %%) switched and %d (%.2f %%) not-switched pixels" % swPrint
        yes_no = raw_input("Do you want to save the image (y/N)?")
        yes_no = yes_no.upper()
        if yes_no == "Y":
            fileName = raw_input("Filename (ext=png): ")
            if len(fileName.split("."))==1:
                fileName = fileName+".png"
            fileName = os.path.join(mainDir,fileName)
            imOut.save(fileName)

            
            
    def imDiffCalculated(self,imageNum,haveColors=True):
        """
        Get the difference in BW between two images imageNum and imageNum+1
        as calculated by the self.colorImage
        """
        if not self.colorImageDone:
            self.checkColorImageDone(ask=False)
        imDC = (self.switchTimesArray==imageNum)*1
        if haveColors:
            imDC = scipy.array(imDC,dtype='int16')
            structure = [[0, 1, 0], [1,1,1], [0,1,0]]
            l, n = nd.label(imDC,structure)
            plt.imshow(l,plt.cm.prism)
        else:
            # Normalize to a BW image
            self.imDiffCalcArray = imDC*255
            scipy.misc.toimage(self.imDiffCalcArray).show()
        return None

    def getDistributions(self,NN=4,log_step=0.2,edgeThickness=1):
        #Define the numer of nearest neighbourg
        if NN==8:
            structure = [[1, 1, 1], [1,1,1], [1,1,1]]
        else:
            structure = [[0, 1, 0], [1,1,1], [0,1,0]]
            if NN!=4:
                print "N. of neibourgh not valid: assuming NN=4"
        # Check if analysis of avalanches has been performed
        if not self.colorImageDone:
            self.checkColorImageDone(ask=False)
        # Select the images having swithing pixels
        # and initialize the distributions
        self.switchTimesArray = np.array(self.switchTimes).reshape((self.dimX, self.dimY))
        n_max = max(self.switchTimes)
        n_min = min(self.switchTimes)
        self.D_avalanches = []
        self.D_cluster = scipy.array([], dtype='int32')
        self.N_cluster = []
        self.dictAxy = {}
        a0 = scipy.array([],dtype='int32')
        #
        # Make a loop to calculate avalanche and clusters for each image
        #
        for imageNum in range(n_min,n_max+1): 
            strOut = 'Analysing image n:  %i\r' % imageNum
            sys.stdout.write(strOut)
            sys.stdout.flush()
            # Select the pixel flipped at the imageNum
            im0 = (self.switchTimesArray==imageNum)*1
            im0 = scipy.array(im0,dtype="int16")
            # Update the list of sizes of the global avalanche (i.e. for the entire image n. imageNum)
            self.D_avalanches.append(scipy.sum(im0))
            # Detect local clusters using scipy.ndimage method
            array_labels, n_labels = nd.label(im0,structure)
            # Make a list the sizes of the clusters
            list_sizes = nd.sum(im0,array_labels,range(1,n_labels+1))
            # Prepare the distributions
            self.D_cluster = scipy.concatenate((self.D_cluster,list_sizes))
            self.N_cluster.append(n_labels)
            # Now find the Axy distributions (A00, A10, etc)
            # First make an array of the edges each cluster touches
            array_Axy = gAL.getAxyLabels(array_labels,self.imageDir,edgeThickness)
            # Note: we can restrict the choice to left and right edges (case of strip) using:
            # array_Axy = [s[:2] for s in array_Axy]
            # Now select each type of cluster ('0000', '0010', etc), make the S*P(S), and calculate the distribution
            array_sizes = scipy.array(list_sizes,dtype='int32')
            for Axy in set(array_Axy):
                sizes = array_sizes[array_Axy==Axy] # Not bad...
                self.dictAxy[Axy] = scipy.concatenate((self.dictAxy.get(Axy,a0),sizes))

        # Calculate and plot the distributions of clusters and avalanches
        D_x, D_y = gLD.logDistribution(self.D_cluster,log_step=log_step,first_point=1.,normed=True)
        P_x, P_y = gLD.logDistribution(self.D_avalanches,log_step=log_step,first_point=1.,normed=True)
        # Plots of the distributions
        plt.figure(1)
        plt.loglog(D_x,D_y,'o', label='cluster')
        plt.loglog(P_x,P_y,'v',label='avalanches')
        plt.legend()
        plt.show()
        # Show the N_clusters vs. size_of_avalanche
        plt.figure(2)
        plt.loglog(self.D_avalanches,self.N_cluster,'o')
        plt.xlabel("Avalanche size")
        plt.ylabel("N. of clusters")
        plt.show()

        

        
    def getDictOfColors(self,outPixels):
        d = {}
        for c in outPixels:
            d[c] = d.get(c,0) +1
        return d


    def getWindowDistributions(self):
        """
        classify windowed avs according to switch times, this outputs distribution for each run
        """
        switchTimesList = self.switchTimes	
        avs_list = set(switchTimesList)
        switchTimesArray = switchTimesList.reshape((500,500)) # reshape array into (x,y) 
        
        for n_av in avs_list:
            single_av = (switchTimesArray==n_avs)*1
            area = np.sum(single_av)
            height = max(np.sum(single_av, axis=0))
            width = max(np.sum(single_av,axis=1))
            
            if area in A_s.keys():
                A_s[area]+=area
            else:
                A_s[area] = area
                
            
            
            # determine how many boundaries each avalanche touches to classify
            
            
        
    # Select dir for analysis: TO IMPROVE
    #mainDir = "/home/gf/meas/Baxrkh/Films/CoFe/20nm/run3_50x/down/"
    #mainDir = "/home/gf/meas/Barkh/Films/FeBSi/50nm/run1/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run1_20x/down"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run9_20x_5ms"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run10_20x_bin1"
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run11_20x_bin1_contrast_diff"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run15_20x_save_to_memory/down"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run32"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/good set 2/run5/"
    #mainDir = "/home/gf/meas/MO/py170/20x/set7"
    #mainDir = "/home/gf/Misure/Alex/Zigzag/samespot/run2"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run22_50x_just_rough/down"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/run23_50x_rough_long"
#mainDir = "/home/gf/meas/Simulation"
#mainDir = "/media/DATA/meas/MO/CoFe 20 nm/10x/good set 2/run7"
##mainDir = "/media/DATA/meas/MO/CoFe 20 nm/5x/set1/run1/"
#mainDir = "/home/gf/meas/Barkh/Films/CoFe/20nm/10x/good set 2/run8/"


#firstImage = 0
#lastImage = None
#images = StackImages(mainDir,sigma=2.5,resize_factor=False,fileType=None,\
                    #imageFirst=firstImage, imageLast=lastImage)
