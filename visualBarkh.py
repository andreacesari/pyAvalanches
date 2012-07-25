import os, sys, glob
import re
import scipy
import scipy.ndimage as nd
import scipy.signal as signal
import scipy.stats.stats
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
#import tables
import Image
import time
import getLogDistributions as gLD
reload(gLD)
import getAxyLabels as gal
reload(gal)
# Load scikits modules if available
try:
    from skimage.filter import tv_denoise
    isTv_denoise = True
except:
    isTv_denoise = False
try:
    import skimage.io as im_io
    
    class Imread_convert():
        def __init__(self, mode):
            self.mode = mode
            
        def __call__(self, f):
            if self.mode != "I;16":
                return im_io.imread(f).astype(np.int16)
            else:
                im = Image.open(f)
                imageList = list(im.getdata())
                sizeX, sizeY = im.size
                return np.asanyarray(imageList).reshape(sizeY, sizeX)

    isScikits = True
except:
    isScikits = False

if isScikits:
    plugins = im_io.plugins()
    keys = plugins.keys()
    #mySeq = ['test','pil', 'matplotlib', 'qt']
    #mySeq = [pl for pl in plugins.keys() if pl in mySeq]
    try:
        im_io.use_plugin('pil', 'imread')
    except:
        print("No plugin available between %s" % str(mySeq))
else:
    print "Scikits.image not available"


filters = {'gauss': nd.gaussian_filter, 'fouriergauss': nd.fourier_gaussian, \
           'median': nd.median_filter, 'wiener': signal.wiener}

if isTv_denoise:
    filters['tv'] = tv_denoise

# Adjust the interpolation scheme to show the images
mpl.rcParams['image.interpolation'] = 'nearest'


class StackImages:
    """
    Load and analyze a sequence of images 
    as a multi-dimensional scipy 3D array.
    The k-th element of the array (i.e. myArray[k])
    is the k-th image of the sequence.
    
    Parameters:
    ----------------
    mainDir : string
        Directory of the image files
    
    pattern : string
        Pattern of the input image files, 
        as for instance "Data1-*.tif"
    
    firstImage, lastImage : int, opt
       first and last image (included) to be loaded
       These numbers refer to the numbers of the filenames 
        
    resize_factor : int, opt
       Reducing factor for the size of raw images
       
    filtering : string, opt
        Apply a filter to the raw images between the following:
        'gauss': nd.gaussian_filter (Default)
        'fouriergauss': nd.fourier_gaussian, 
        'median': nd.median_filter, 
        'tv': tv_denoise, 
        'wiener': signal.wiener
        
    sigma : scalar or sequence of scalars, required with filtering
       for 'gauss': standard deviation for Gaussian kernel.
       for 'fouriergauss': The sigma of the Gaussian kernel.
       for 'median': the size of the filter
       for 'tv': denoising weight
       for 'wiener': A scalar or an N-length list giving the size of the Wiener filter
       window in each dimension.
    """
        
    def __init__(self,mainDir,pattern, resize_factor=None, \
                 firstImage=None, lastImage=None,\
                 filtering=None, sigma=None):
        # Initialize variables
        self._mainDir = mainDir
        self._colorImage = None
        self._koreanPalette = None
        self._isColorImage = False
        self._isSwitchAndStepsDone = False
        self._switchTimes = None
        self._threshold = 0
        self._figTimeSeq = None
        self.figDiffs = None
        self._figHistogram = None
        self._figColorImage = None
        self._figColorImage2 = None
        if lastImage == None:
            lastImage = -1
        # Make a kernel as a step-function
        self.kernel = np.array([-1]*(5) +[1]*(5)) # Good for Black_to_White change of grey scale
        self.kernel0 = np.array([-1]*(5) +[0] + [1]*(5)) 
        if not os.path.isdir(mainDir):
            print("Please check you dir %s" % mainDir)
            print("Path not found")
            sys.exit()
        # Collect the list of images in mainDir
        s = "(%s|%s)" % tuple(pattern.split("*"))
        patternCompiled = re.compile(s)
        # Load all the image filenames
        imageFileNames = sorted(glob.glob1(mainDir, pattern))
        if not len(imageFileNames):
            print "ERROR, no images in %s" % mainDir
            sys.exit()
        else:
            print "Found %d images in %s" % (len(imageFileNames), mainDir)
        # Search the number of all the images given the pattern above
        _imageNumbers = [int(patternCompiled.sub("",fn)) for fn in imageFileNames]
        # Search the indexes where there are the first and the last images to be loaded
        if firstImage is None:
            firstImage = _imageNumbers[0]
        if lastImage < 0:
            lastImage = len(_imageNumbers)  + lastImage + firstImage
        try:
            indexFirst, indexLast = _imageNumbers.index(firstImage), _imageNumbers.index(lastImage)
        except:
            i0, i1 = _imageNumbers[0], _imageNumbers[-1]
            print("Error: range of the images is %s-%s (%s-%s chosen)" % (i0,i1,firstImage, lastImage))
            sys.exit()
        # Save the list of numbers of the images to be loaded
        self.imageNumbers = _imageNumbers[indexFirst:indexLast+1]
        self.imageIndex = []
        print "First image: %s" % imageFileNames[indexFirst]
        print "Last image: %s" % imageFileNames[indexLast]
        # Check the mode of the images
        imageMode = Image.open(os.path.join(mainDir, imageFileNames[indexFirst])).mode
        imread_convert = Imread_convert(imageMode)
        # Load the images
        print "Loading images: "
        load_pattern = [os.path.join(mainDir,ifn) for ifn in imageFileNames[indexFirst:indexLast+1]]
        
        if isScikits:
            imageCollection = im_io.ImageCollection(load_pattern, load_func=imread_convert)
        else:
            sys.exit()
        if filtering:
            filtering = filtering.lower()
            if filtering not in filters:
                print "Filter not available"
                sys.exit()
            else:
                print "Filter: %s" % filtering
                if filtering == 'wiener':
                    sigma = [sigma, sigma]
                self.Array = np.dstack([np.int16(filters[filtering](im,sigma)) for im in imageCollection])
        else:
            self.Array = np.dstack([im for im in imageCollection])
        self.shape = self.Array.shape 
        self.dimX, self.dimY, self.n_images = self.shape
        print "%i image(s) loaded, of %i ximport skimage.io as io %i pixels" % (self.n_images, self.dimX, self.dimY)
        # Check for the grey direction
        grey_first_image = scipy.mean(self.Array[:,:,0].flatten())
        grey_last_image = scipy.mean(self.Array[:,:,-1].flatten())
        print "grey scale: %i, %i" % (grey_first_image, grey_last_image)
        if grey_first_image > grey_last_image:
            self.kernel = -self.kernel
            self.kernel0 = -self.kernel0

    def __get__(self):
        return self.Array
        
    def __getitem__(self,n):
        """Get the n-th image"""
        index = self._getImageIndex(n)
        if index is not None:
            return self.Array[:,:,index]

    def _getImageIndex(self,n):
        """
        check if image number n has been loaded
        and return the index of it in the Array
        """
        ns = self.imageNumbers
        try:
            return ns.index(n)
        except:
            print "Image number %i is out of the range (%i,%i)" % (n, ns[0], ns[-1])
            return None
        
    def showRawImage(self, imageNumber, plugin='mpl'):
        """
        showImage(imageNumber)
        
        Show the n-th image where n = image_number
        
        Parameters:
        ---------------
        imageNumber : int
            Number of the image to be shown.
            
        plugin : str, optional
        Use a plugin to show an image (default: matplotlib)
        """
        n = self._getImageIndex(imageNumber)
        if n is not None:
            im = self[imageNumber]
            if plugin == 'mpl':
                plt.imshow(im, plt.cm.gray)
            else:
                im_io.imshow(self[imageNumber])
        
    def _getWidth(self):
        try:
            width = self.width
        except:
            self.width = 'all'
            print("Warning: the levels are calculated over all the points of the sequence")
        return self.width
        
    def _getLevels(self, pxTimeSeq, switch, kernel='step'):
        """
        _getLevels(pxTimeSeq, switch, kernel='step')
        
        Internal function to calculate the gray level before and 
        after the switch of a sequence, using the kernel 
        
        Parameters:
        ---------------
        pxTimeSeq : list
            The sequence of the gray level for a given pixel.
        switch : number, int
            the position of the switch as calculated by getSwitchTime
        kernel : 'step' or 'zero'
           the kernel of the step function

        Returns:
        -----------
        levels : tuple
           Left and right levels around the switch position
        """
        width = self._getWidth()
            
        # Get points before the switch
        if width == 'small': 
            halfWidth = len(self.kernel)/2
            lowPoint = switch - halfWidth - 1*(kernel=='zero')
            if lowPoint < 0:
                lowPoint = 0
            highPoint = switch + halfWidth
            if highPoint > len(pxTimeSeq):
                highPoint = len(pxTimeSeq)
        elif width == 'all':
            lowPoint, highPoint = 0, len(pxTimeSeq)
        else:
            print 'Method not implement yet'
            return None
        leftLevel = np.int(np.mean(pxTimeSeq[lowPoint:switch - 1*(kernel=='zero')])+0.5)
        rigthLevel = np.int(np.mean(pxTimeSeq[switch:highPoint])+0.5)
        levels = leftLevel, rigthLevel 
        return levels
    
    
    def pixelTimeSequence(self,pixel=(0,0)):
        """
        pixelTimeSequence(pixel)
        
        Returns the temporal sequence of the gray level of a pixel
        
        Parameters:
        ---------------
        pixel : tuple
           The (x,y) pixel of the image, as (row, column)
        """
        x,y = pixel
        return self.Array[x,y,:]
        
    def showPixelTimeSequence(self,pixel=(0,0),newPlot=False):
        """
        pixelTimeSequenceShow(pixel)
        
        Plot the temporal sequence of the gray levels of a pixel;
        
        Parameters:
        ---------------
        pixel : tuple
            The (x,y) pixel of the image, as (row, column)
        newPlot : bool
            Option to open a new frame or use the last one
        """
        width = self._getWidth()
        # Plot the temporal sequence first
        pxt = self.pixelTimeSequence(pixel)
        if not self._figTimeSeq or newPlot==True:
            self._figTimeSeq = plt.figure()
        else:
            self._figTimeSeq
        plt.plot(self.imageNumbers,pxt,'-o')
        # Add the two kernels function
        kernels = [self.kernel, self.kernel0]
        for k,kernel in enumerate(['step','zero']):	
            switch, (value_left, value_right) = self.getSwitchTime(pixel, useKernel=kernel)
            print "switch %s, Kernel = %s" % (kernel, switch)
            print ("gray level change at switch = %s") % abs(value_left-value_right)
            if width == 'small':
                halfWidth = len(kernels[k])/2
                x0,x1 = switch - halfWidth - 1*(k==1), switch + halfWidth
                x = range(x0,x1)
                n_points_left = halfWidth
                n_points_rigth = halfWidth
            elif width=='all':
                #x = range(len(pxt))
                x = self.imageNumbers
                n_points_left = switch - 1 * (k==1)
                n_points_rigth = len(pxt) - switch
            y = n_points_left * [value_left] + [(value_left+value_right)/2.] * (k==1) + n_points_rigth * [value_right]
            
            plt.plot(x,y)
        plt.draw()
        plt.show()
        
    def getSwitchTime(self, pixel=(0,0), useKernel='step', method='convolve1d'):
        """
        getSwitchTime(pixel, useKernel='step', method="convolve1d")
        
        Return the position of a step in a sequence
        and the left and the right values of the gray level (as a tuple)
        
        Parameters:
        ---------------
        pixel : tuple
            The (x,y) pixel of the image, as (row, column).
        useKernel : string
            step = [1]*5 +[-1]*5
            zero = [1]*5 +[0] + [-1]*5
            both = step & zero, the one with the highest convolution is chosen
        method : string
            For the moment, only the 1D convolution calculation
            with scipy.ndimage.convolve1d is available
        """
        pxTimeSeq = self.pixelTimeSequence(pixel)
        if method == "convolve1d":
            if useKernel == 'step' or useKernel == 'both':
                convolution_of_stepKernel = nd.convolve1d(pxTimeSeq,self.kernel)
                minStepKernel = convolution_of_stepKernel.min()
                switchStepKernel = convolution_of_stepKernel.argmin() +1
                switch = switchStepKernel
                kernel_to_use = 'step'
            if useKernel == 'zero' or useKernel == 'both':
                convolution_of_zeroKernel = nd.convolve1d(pxTimeSeq,self.kernel0)
                minZeroKernel = convolution_of_zeroKernel.min()
                switchZeroKernel = convolution_of_zeroKernel.argmin() + 1
                switch = switchZeroKernel
                kernel_to_use = 'zero'
            if useKernel == 'both':
                if minStepKernel <= minZeroKernel:
                    switch = switchStepKernel
                    kernel_to_use = 'step'
                else:
                    switch = switchZeroKernel
                    kernel_to_use = 'zero'
                    #leftLevel = np.int(np.mean(pxTimeSeq[0:switch])+0.5)
                    #rightLevel = np.int(np.mean(pxTimeSeq[switch+1:])+0.5)
                    #middle = (leftLevel+rightLevel)/2
                    #rightLevelStep = np.int(np.mean(pxTimeSeq[switchStepKernel+1:])+0.5)
                    #if abs(pxTimeSeq[switch]-middle)>abs(pxTimeSeq[switch]-rightLevelStep):
                        #switch = switchStepKernel                    
                    #switch = (switch-1)*(pxTimeSeq[switch]<middle)+switch*(pxTimeSeq[switch]>=middle)
                #switch = switchStepKernel * (minStepKernel<=minZeroKernel/1.1) + switchZeroKernel * (minStepKernel >minZeroKernel/1.1)
        else:
            raise RuntimeError("Method not yet implemented")            
        levels = self._getLevels(pxTimeSeq, switch, kernel_to_use)
        # Now redefine the switch using the correct image number
        switch = self.imageNumbers[switch]
        return switch, levels

    def _imDiff(self, imNumbers, invert=False):
        """Properly rescaled difference between images

        Parameters:
        ---------------
        imNumbers : tuple
        the numbers the images to subtract
        invert : bool
        Invert black and white grey levels
        """
        i, j = imNumbers
        try:
            im = self[i]-self[j]
        except:
            return 
        if invert:
            im = 255 - im
        imMin = scipy.amin(im)
        imMax = scipy.amax(im)
        im = scipy.absolute(im-imMin)/float(imMax-imMin)*255
        return scipy.array(im,dtype='int16')

    def showTwoImageDifference(self, imNumbers, invert=False):
        """Show the output of self._imDiff
        
        Parameters:
        ---------------
        imNumbers : tuple
        the numbers of the two images to subtract
        
        invert : bool, opt
        Invert the gray level black <-> white
        """
        if type(invert).__name__ == 'int':
            imNumbers = imNumbers, invert
            print("Warning: you should use a tuple as image Numbers")
            
        try:
            plt.imshow(self._imDiff(imNumbers, invert),plt.cm.gray)
        except:
            return
        
    def imDiffSave(self,imNumbers='all', invert=False, mainDir=None):
        """
        Save the difference(s) between a series of images
        
        Parameters:
        ---------------
        imNumbers : tuple or string
        the numbers of the images to subtract
        * when 'all' the whole sequence of differences is saved
        * when a tuple of two number (i.e., (i, j), 
        all the differences of the images between i and j (included)
        are saved
        """
        if mainDir == None:
            mainDir = self._mainDir
        dirSeq = os.path.join(mainDir,"Diff")
        if not os.path.isdir(dirSeq):
            os.mkdir(dirSeq)
        if imNumbers == 'all':
            imRange = self.imageNumbers[:-1]
        else:
            im0, imLast = imNumbers
            imRange = range(im0, imLast)
            if im0 >= imLast:
                print "Error: sequence not valid"
                return
        for i in imRange:
            im = self._imDiff((i+1,i))
            imPIL = scipy.misc.toimage(im)
            fileName = "imDiff_%i_%i.tif" % (i+1,i)
            print fileName
            imageFileName = os.path.join(dirSeq, fileName)
            imPIL.save(imageFileName)

    def getSwitchTimesAndSteps(self):
        """
        Calculate the switch times and the gray level changes
        for each pixel in the image sequence.
        It calculates:
        self._switchTimes
        self._switchSteps
        """
        switchTimes = []
        switchSteps = []
        startTime = time.time()
        # ####################
        # TODO: make here a parallel calculus
        for x in range(self.dimX):
            # Print current row
            if not (x+1)%10:
                strOut = 'Analysing row:  %i/%i on %f seconds\r' % (x+1, self.dimX, time.time()-startTime)
                sys.stderr.write(strOut)
                #sys.stdout.flush()
                startTime = time.time()
            for y in range(self.dimY):
                switch, levels = self.getSwitchTime((x,y))
                grayChange = np.abs(levels[0]- levels[1])
                if switch == 0: # TODO: how to deal with steps at zero time
                    print x,y
                switchTimes.append(switch)
                switchSteps.append(grayChange)
        print "\n"
        self._switchTimes = np.asarray(switchTimes)
        self._switchSteps = np.asarray(switchSteps)
        self._isColorImage = True
        self._isSwitchAndStepsDone = True
        return

    def _getSwitchTimesArray(self, threshold=0, isFirstSwitchZero=False, fillValue=-1):
        """
        _getSwitchTimesArray(threshold=0)
        
        Returns the array of the switch times
        considering a threshold in the gray level change at the switch
        
        Parameters:
        ----------------
        threshold : int
            The miminum value of the gray level change at the switch
        isFirstSwitchZero : bool
            Put the first switch equal to zero, useful to set the colors 
            in a long sequence of images where the first avalanche 
            occurs after many frames
        fillValue : number, int
            The value to set in the array for the non-switching pixel (below the threshold)
            -1 is use as the last value of array when used as index (i.e. with colors)
        """
        if not threshold:
            threshold = 0
        self.isPixelSwitched = self._switchSteps >= threshold    
        maskedSwitchTimes = ma.array(self._switchTimes, mask = ~self.isPixelSwitched)
        # Move to the first switch time if required
        if isFirstSwitchZero:
            maskedSwitchTimes = maskedSwitchTimes - self.min_switch
        # Set the non-switched pixels to use the last value of the pColor array, i.e. noSwitchColorValue
        switchTimes = maskedSwitchTimes.filled(fillValue) # Isn't it fantastic?
        return switchTimes

    def _getKoreanColors(self, switchTime, n_images=None):
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
    
    def _isColorImageDone(self,ask=True):
        print "You must first run the getSwitchTimesAndSteps script: I'll do that for you"
        if ask:
            yes_no = raw_input("Do you want me to run the script for you (y/N)?")
            yes_no = yes_no.upper()
            if yes_no != "Y":
                return
        self.getSwitchTimesAndSteps()
        return

    def getColorImage(self, threshold=None, palette='korean', noSwitchColor='black'):
        """
        Calculate the color Image using the output of getSwitchTimesAndSteps
        
        Parameters:
        ---------------
        threshold: int, opt
        Set the minimim value of the gray level change at the switch to 
        consider the pixel as 'switched', i.e. belonging to an avalanche
        This value is set as a class variable from here on.
        Rerun self.getColorImage to change it
        
        Results:
        ----------
        self._switchTimes2D as a 2D array of the switchTime 
        with steps >= threshold, and first image number set to 0
        """
        if not threshold:
            threshold = 0
        self._threshold = threshold
        if not self._isSwitchAndStepsDone:
            self._isColorImageDone(ask=False)
            
        self.min_switch = np.min(self._switchTimes)
        self.max_switch = np.max(self._switchTimes)
        print "Avalanches occur between frame %i and %i" % (self.min_switch, self.max_switch)
        nImagesWithSwitch = self.max_switch - self.min_switch + 1
        print "Gray changes are between %s and %s" % (min(self._switchSteps), max(self._switchSteps))

        # Calculate the colours, considering the range of the switch values obtained 
        if self._koreanPalette is None:
            # Prepare the Korean Palette
            self._koreanPalette = np.array([self._getKoreanColors(i, nImagesWithSwitch) for i in range(nImagesWithSwitch)])            
            
        if palette == 'korean':
            pColor = self._koreanPalette
        elif palette == 'randomKorean':
            pColor = np.random.permutation(self._koreanPalette)
        elif palette == 'random':
            pColor = np.random.randint(0, 256, self._koreanPalette.shape)
        elif palette == 'randomHue': 
            # Use equally spaced colors in the HUE weel, and
            # then randomize
            pColor = [hsv_to_rgb(j/float(nImagesWithSwitch),1, np.random.uniform(0.75,1)) for j in range(nImagesWithSwitch)]
            pColor = np.random.permutation(pColor)
        if noSwitchColor == 'black':
            noSwitchColorValue = 3*[0]
        elif noSwitchColor == 'white':
            noSwitchColorValue = 3*[255]
        self._pColors = np.concatenate(([noSwitchColorValue], pColor))/255.
        self._colorMap = mpl.colors.ListedColormap(self._pColors, 'pColorMap')
        #Calculate the switch time Array (2D) considering the threshold and the start from zero
        self._switchTimes2D = self._getSwitchTimesArray(threshold, True, -1).reshape(self.dimX, self.dimY)
        return

    
    def showColorImage(self, threshold=None, palette='random', noSwitchColor='black', ask=False):
        """
        showColorImage([threshold=None, palette='random', noSwitchColor='black', ask=False])
        
        Show the calculated color Image of the avalanches.
        Run getSwitchTimesAndSteps if not done before.
        
        Parameters
        ---------------
        threshold: integer, optional
            Defines if the pixel switches when gray_level_change >= threshold
        palette: string, required, default = 'korean'
            Choose a palette between 'korean', 'randomKorean', and 'random'
            'randomKorean' is a random permutation of the korean palette
            'random' is calculated on the fly, so each call of the method gives different colors
        noSwithColor: string, optional, default = 'black'
            background color for pixels having gray_level_change below the threshold
        """
        # Calculate the Color Image
        self.getColorImage(threshold, palette, noSwitchColor)
        # Prepare to plot
        self._figColorImage = self._plotColorImage(self._switchTimes2D, self._colorMap, self._figColorImage)
        # Plot the histogram
        self._plotHistogram(self._switchTimes)
        # Count the number of the switched pixels
        switchPixels = np.sum(self.isPixelSwitched)
        totNumPixels = self.dimX * self.dimY
        noSwitchPixels = totNumPixels - switchPixels
        swPrint = (switchPixels, switchPixels/float(totNumPixels)*100., noSwitchPixels, noSwitchPixels/float(totNumPixels)*100.)
        print "There are %d (%.2f %%) switched and %d (%.2f %%) not-switched pixels" % swPrint
        #yes_no = raw_input("Do you want to save the image (y/N)?")
        #yes_no = yes_no.upper()
        #if yes_no == "Y":
            #fileName = raw_input("Filename (ext=png): ")
            #if len(fileName.split("."))==1:
                #fileName = fileName+".png"
            #fileName = os.path.join(self._mainDir,fileName)
            #imOut = scipy.misc.toimage(self._colorImage)
            #imOut.save(fileName)
    
    def _call_lambda(self,x,y):
        x, y = int(y+0.5), int(x+.5)
        if x >= 0 and x < self.dimX and y >= 0 and y < self.dimY:
            index = x * self.dimY + y
            s = "pixel (%i,%i) - switch at: %i, gray step: %i" % \
                (x, y, self._switchTimes[index], self._switchSteps[index])
            return s
        else:
            return None
            
    def _plotColorImage(self, data, colorMap, fig=None):
        if fig == None:
            fig = plt.figure()
            fig.set_size_inches(7,6,forward=True)
            ax = fig.add_subplot(1,1,1)
        else:
            plt.figure(fig.number)
            ax = fig.gca()
        ax.format_coord = lambda x,y: self._call_lambda(x, y)
        #ax.set_title(title)
        # Sets the limits of the image
        extent = 0, self.dimY - 1, self.dimX - 1, 0
        plt.imshow(data, colorMap, norm=mpl.colors.NoNorm(), extent = extent)
        return fig
    
    def _plotHistogram(self, data):
        rng = np.arange(self.min_switch-0.5, self.max_switch+0.5)
        if self._figHistogram is None:
            self._figHistogram = plt.figure()
            #ax = self._figHistogram.add_subplot(1,1,1)
        else:
            plt.figure(self._figHistogram.number)
            plt.clf()
        N, bins, patches = plt.hist(data, rng)
        ax = plt.gca()
        ax.format_coord =  lambda x,y : "image Number: %i, avalanche size (pixels): %i" % (int(x+0.5), int(y+0.5))
        plt.xlabel("image number")
        plt.ylabel("Avalanche size (pixels)")
        for i in range(len(N)):
            patches[i].set_color((tuple(self._pColors[i])))
        plt.show()    

    def saveColorImage(self,fileName,threshold=None, palette='korean',noSwitchColor='black'):
        """
        saveColorImage(fileName, threshold=None, palette='korean',noSwitchColor='black')
        
        makes color image and saves
        """
        self._colorImage = self.getColorImage(threshold, palette,noSwitchColor)
        imOut = scipy.misc.toimage(self._colorImage)
        imOut.save(fileName)
        
    def saveImage(self, figureNumber):
        """
        generic method to save an image or a plot in the figure(figureNumber)
        """
        filename = raw_input("file name (.png for images)? ")
        fig = plt.figure(figureNumber)
        ax = fig.gca()
        if len(ax.get_images()):
            im = ax.get_images()[-1]
            name, ext = os.path.splitext(filename)
            if ext != ".png":
                filename = name +".png"
            filename = os.path.join(self._mainDir, filename)
            im.write_png(filename)
        else:
            filename = os.path.join(self._mainDir, filename)
            fig.save(filename)
        
                
    def imDiffCalculated(self,imageNum,haveColors=True):
        """
        Get the difference in BW between two images imageNum and imageNum+1
        as calculated by the self._colorImage
        """
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        imDC = (self._switchTimes2D==imageNum)*1
        if haveColors:
            imDC = scipy.array(imDC,dtype='int16')
            structure = [[0, 1, 0], [1,1,1], [0,1,0]]
            l, n = nd.label(imDC,structure)
            im_io.imshow(l,plt.cm.prism)
        else:
            # Normalize to a BW image
            self.imDiffCalcArray = imDC*255
            scipy.misc.toimage(self.imDiffCalcArray).show()
        return None
    
    def showRawAndCalcImages(self, n, threshold=0):
        if self._switchTimes == None:
            print("Need to calculate the color image first")
            return
        if n in self._switchTimes:
            fig = plt.figure()
            fig.set_size_inches(12,6,forward=True)
            plt.subplot(1,2,1)
            plt.imshow(self._imDiff((n,n-1)),plt.cm.gray)
            plt.title("Fig. %s, Original" % n)
            plt.grid(color='blue', ls="-")
            plt.subplot(1,2,2)
            switchTimes_images = self._getSwitchTimesArray(threshold, fillValue=0).reshape(self.dimX, self.dimY) == n
            cl = self._pColors[n - self.min_switch]
            myMap = mpl.colors.ListedColormap([(0,0,0),cl],'mymap',2)
            plt.imshow(switchTimes_images, myMap)
            plt.title("Fig. %s, Calculated" % n)
            plt.grid(color='blue',ls="-")
        else:
            print "No switch there"
        return
            
    def ghostbusters(self, clusterThreshold = 15, showImages=False, imageNumber=None):
        """
        Find the presence of 'ghost' images, 
        given by not fully-resolved avalanches.
        Calculates the number of clusters for each avalanche,
        and see if it larger than the threshold.
        *** Automatic check
        If it is, checks if also the following image has an avalanche 
        with a large number of cluster
        In positive, join the two avalanches, and check the number 
        of clusters again. If it smaller than the threshold, 
        the switch time is updated.
        *** Manual check
        If showImages is enabled, the user must manually set the images to join
        *** If the imageNumber is given,
        the method works only on that frame and the following one.
        This is used to manually check two frames and joint them
        
        Parameters
        ---------------
        clusterThreshold : int
        Minimum number of clusters to consider the avalanche as 'spongy'
        
        showImages : bool
        If True, show all the 'spongy' images and their joint one
        """
        joinImages = False
        if not self._isColorImage:
            print("This is available only after the color image is done")
            return
        if showImages:
            figCluster = plt.figure()
            figCluster.set_size_inches(12, 8, forward=True)
            i0 = [0,1,1] # Index of the first raw image
            i1 = [-1,0,-1] # Index of the second raw image
        n_of_images_with_ghosts = []
        images_with_ghosts = {}
        structure = [[1, 1, 1], [1,1,1], [1,1,1]]
        if imageNumber:
            iterator = np.asarray([imageNumber, imageNumber+1]) - self.min_switch
            clusterThreshold = 0
        else:
            iterator = np.unique(self._switchTimes2D)
        # Calculates the set of switches
        for imageNumber0 in iterator:
            im0 = (self._switchTimes2D==imageNumber0)*1
            im0 = scipy.array(im0, dtype="int16")
            array_labels, n_clusters = nd.label(im0, structure)
            if n_clusters >= clusterThreshold:
                imageNumber = imageNumber0 + self.min_switch
                n_of_images_with_ghosts.append(imageNumber)
                images_with_ghosts[imageNumber] = array_labels, n_clusters
        # Now evaluate spongy avalanches and check if number of clusters is reduced
        # Let us do it first on consecutive images belonging to n_of_images_with_ghosts
        gh = scipy.asarray(n_of_images_with_ghosts)
        # Consider consecutive images only
        ghosts_images = gh[gh[1:] == gh[:-1]+1]
        if len(ghosts_images) == 0:
            print("Warning, no images to consider")
            return
        for ghi in ghosts_images:
            image1, n1 = images_with_ghosts[ghi]
            image2, n2 = images_with_ghosts[ghi+1]
            new_array = scipy.array(image1+image2, dtype="int16")
            image3, n3 = nd.label(new_array, structure)
            if showImages:
                for i, results in enumerate(zip([image1, image2, image3],[n1, n2, n3])):
                    im, clusters = results
                    plt.subplot(2, 3, i+1)
                    # Prepare the palette, from red to magenta (see hue weel for details)
                    myPalette = [(0,0,0)] + [hsv_to_rgb(j/float(clusters),1,1) for j in range(clusters)]
                    plt.imshow(im, mpl.colors.ListedColormap(myPalette))
                    imageNum = str(ghi+i)*(i<2) + (i==2)*"joint"
                    plt.title("Image: %s, N. clusters: %i" % (imageNum, clusters))
                    plt.subplot(2, 3, i+4)
                    plt.imshow(self._imDiff((ghi+i0[i],ghi+i1[i])), plt.cm.gray)
                y_n = raw_input("Join these avalanches from image %i and %i? (y/N)" % (ghi, ghi+1))
                y_n = y_n.upper()
                if y_n in ["Y", "YES"]:
                    joinImages = True
                else:
                    joinImages = False
            if (n3 < clusterThreshold and not showImages) or (showImages and joinImages):
                print("Joining images %i and %i" % (ghi, ghi + 1))
                whereChange = self._switchTimes==ghi
                # Update the 'untouched' array of switch times
                self._switchTimes[whereChange] = ghi + 1
                # Update the array with threshold and zero time at the beginning
                self._switchTimes2D[whereChange.reshape(self.dimX, self.dimY)] = ghi + 1 - self.min_switch
        # Add the image without ghosts to the original one
        self._figColorImage = self._plotColorImage(self._switchTimes2D, self._colorMap, fig=self._figColorImage)
        self._plotHistogram(self._switchTimes)
        
    def manualGhostbuster(self):
        """
        Manually adjust spongy avalanches by looking the color image
        The raw and the calculated image are presented
        """
        if not self._isColorImage:
            print("This is available only after the color image is done")
            return
        while True:
            imageNumber = raw_input("Number of the image to join with its next (Return to exit): ")
            if imageNumber is not "":
                imageNumber = int(imageNumber)
                self.ghostbusters(0, True, imageNumber)
            else:
                return

            
    def _getImageDirection(self, threshold=None):
        """
        _getImageDirection(threshold=None)
        
        Returns the direction of the sequence of avalanches as: 
        "Top_to_bottom","Left_to_right", "Bottom_to_top","Right_to_left"
        
        Parameters:
        ----------------
        threshold : int
            Minimum value of the gray level change to conisider
            a pixel as part of an avalanche (i.e. it is switched)
        """
        # Top, left, bottom, rigth
        imageDirections=["Top_to_bottom","Left_to_right", "Bottom_to_top","Right_to_left"]
        # Check if color Image is available
        if not self._isColorImage:
            self._isColorImageDone(ask=False)        
        switchTimesMasked = self._switchTimes2D
        pixelsUnderMasks = []
        # first identify first 10 avalanches of whole image
        firstAvsList = np.unique(self._switchTimes2D)[:11]
        # Prepare the mask
        m = np.ones((self.dimX, self.dimY))
        # Top mask
        mask = np.rot90(np.triu(m)) * np.triu(m)
        top = switchTimesMasked * mask
        pixelsUnderMasks.append(sum([np.sum(top==elem) for elem in firstAvsList]))
        # Now we need to rotate the mask
        for i in range(3):
            mask = np.rot90(mask)
            top = switchTimesMasked * mask
            pixelsUnderMasks.append(sum([np.sum(top==elem) for elem in firstAvsList]))
        max_in_mask = scipy.array(pixelsUnderMasks).argmax()
        return imageDirections[max_in_mask]
    
    
    def getDistributions(self, NN=8, log_step=0.2, edgeThickness=1, fraction=0.01):
        """
        Calculates the distribution of avalanches and clusters
        
        Parameters:
        ---------------
        NN : int
        No of Nearest Neighbours around a pixel to consider two clusters
        as touching or not
        
        log_step: float
        The step in log scale between points in the log-log distribution.
        For instance, 0.2 means 5 points/decade
        
        edgeThickness : int
        No of pixels for each edge to consider as the frame of the image 
        
        fraction : float
        This is the minimum fraction of the size of the avalanche/cluster inside
        an edge (of thickness edgeThickness) with sets the avalanche/cluster
        as touching
        """
        # Check if analysis of avalanches has been performed
        if not self._isColorImage:
            self._isColorImageDone(ask=False)
        # Initialize variables
        self.D_avalanches = []
        self.D_cluster = scipy.array([], dtype='int32')
        #self.N_cluster = {}
        self.N_cluster = []
        self.dictAxy = {}
        self.dictAxy['aval'] = {}
        self.dictAxy['clus'] = {}
        a0 = scipy.array([],dtype='int32')
        #Define the number of nearest neighbourg
        if NN==8:
            structure = [[1, 1, 1], [1,1,1], [1,1,1]]
        else:
            structure = [[0, 1, 0], [1,1,1], [0,1,0]]
            if NN!=4:
                print "N. of neibourgh not valid: assuming NN=4"
        # Find the direction of the avalanches (left <-> right, top <-> bottom)
        self.imageDir = self._getImageDirection(self._threshold)
        print self.imageDir
        #
        # Make a loop to calculate avalanche and clusters for each image
        #
        images = np.unique(self._switchTimes2D)
        for imageNum in images: 
            strOut = 'Analysing image n:  %i\r' % (imageNum + self.min_switch)
            sys.stderr.write(strOut)
            #sys.stdout.flush()
            # Select the pixel flipped at the imageNum
            im0 = (self._switchTimes2D == imageNum) * 1
            im0 = scipy.array(im0, dtype="int16")
            # Update the list of sizes of the global avalanche (i.e. for the entire image imageNum)
            avalanche_size = scipy.sum(im0)
            self.D_avalanches.append(avalanche_size)
            # Find how many edges this avalanche touches
            Axy = gal.getAxyLabels(im0, self.imageDir, edgeThickness)
            Axy = Axy[0] # There is only one value for the whole image
            # Update the dictionary of the avalanches
            self.dictAxy['aval'][Axy] = scipy.concatenate((self.dictAxy['aval'].get(Axy,a0), [avalanche_size]))
            # 
            # Now move to cluster distributions
            #
            # Detect local clusters using scipy.ndimage method
            array_labels, n_labels = nd.label(im0, structure)
            # Make a list the sizes of the clusters
            list_clusters_sizes = nd.sum(im0, array_labels, range(1, n_labels+1))
            # Update the distributions
            self.D_cluster = scipy.concatenate((self.D_cluster, list_clusters_sizes))
            #self.N_cluster[avalanche_size] = scipy.concatenate((self.N_cluster.get(avalanche_size, a0), [n_labels]))
            self.N_cluster.append(n_labels)
            # Now find the Axy distributions (A00, A10, etc)
            # First make an array of the edges each cluster touches
            array_Axy = gal.getAxyLabels(array_labels, self.imageDir, edgeThickness)
            # Note: we can restrict the choice to left and right edges (case of strip) using:
            # array_Axy = [s[:2] for s in array_Axy]
            # Now select each type of cluster ('0000', '0010', etc), make the S*P(S), and calculate the distribution
            array_cluster_sizes = scipy.array(list_clusters_sizes, dtype='int32')
            for Axy in np.unique(array_Axy):
                sizes = array_cluster_sizes[array_Axy==Axy] # Not bad...
                self.dictAxy['clus'][Axy] = scipy.concatenate((self.dictAxy['clus'].get(Axy,a0), sizes))
        print()
        print("Done")
        # Calculate and plot the distributions of clusters and avalanches
        D_x, D_y = gLD.logDistribution(self.D_cluster, log_step=log_step, first_point=1., normed=True)
        P_x, P_y = gLD.logDistribution(self.D_avalanches, log_step=log_step, first_point=1., normed=True)
        # Plots of the distributions
        plt.figure()
        plt.loglog(D_x,D_y,'o', label='cluster')
        plt.loglog(P_x,P_y,'v',label='avalanches')
        plt.legend()
        plt.show()
        # Show the N_clusters vs. size_of_avalanche
        plt.figure()
        clusterArray = np.array(zip(self.D_avalanches, self.N_cluster))
        sizeCluster, nClusters = gLD.averageLogDistribution(clusterArray, log_step=log_step, first_point=1.)
        plt.loglog(sizeCluster, nClusters,'o')
        plt.xlabel("Avalanche size")
        plt.ylabel("N. of clusters")
        plt.show()
    

if __name__ == "__main__":
    mainDir, pattern, firstImage, lastImage = "/media/DATA/meas/MO/CoFe/50nm/20x/run5/", "Data1-*.tif", 280, 1029
    mainDir, pattern, firstImage, lastImage = "/media/DATA/meas/Barkh/Films/CoFe/50nm/20x/run5/", "Data1-*.tif", 280, 1029    
    #mainDir, pattern, firstImage, lastImage  = "/media/DATA/meas/MO/Picostar/orig", "B*.TIF", 0, 99
    
    imArray = StackImages(mainDir, pattern, resize_factor=False,\
                             filtering='gauss', sigma=1.5,\
                             firstImage=firstImage, lastImage=lastImage)

    imArray.width='small'
    imArray.useKernel = 'step'
