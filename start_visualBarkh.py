# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:06:23 2011

@author: gf
"""

import visualBarkh as vB
reload(vB)

if __name__ == "__main__":
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/50nm/run2/"
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
    #mainDir = "/home/gf/meas/Barkh/Films/CoFe/50nm/run2/"
    #mainDir, firstImage, lastImage, filtering, sigma, pattern= ("/media/DATA/meas/MO/CoFe/50nm/20x/run5/", 250, 1050, 'gauss', 1.5, "Data1-*.tif")
    #mainDir, firstImage, lastImage, filtering, sigma, pattern= ("/media/DATA/meas/MO/CoFe/50nm/20x/run6/", 470, 1431, 'gauss', 1.5, "Data1-*.tif")
    mainDir, firstImage, lastImage, filtering, sigma, pattern= ("/media/DATA/meas/MO/CoFe/50nm/20x/run7/", 120, 1180, 'gauss', 1.5, "Data1-*.tif")
    
    imArray = vB.StackImages(mainDir,pattern, filtering=filtering, sigma=sigma,\
                             firstImage=firstImage, lastImage=lastImage)

    imArray.width='small'
    imArray.useKernel = 'step'
