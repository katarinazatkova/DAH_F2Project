# f2_project.py
# Project F2: Make Accurate Measurements of Particle Masses

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from scipy.stats import norm
from iminuit import Minuit
from iminuit.cost import UnbinnedNll


class F2Project(object):
    
    """
        F2 project class.
    
    Args:

    """

    
    def __init__(self, xmass):
        
        self.xmass = xmass 
        self.xmass_min = 1800
        self.xmass_max = 2050

    
    def part_1(self):

        def norm_exp(self, x, tau):
            exp_norm_factor = np.exp(-self.xmass_min/tau) - np.exp(-self.xmass_max/tau)
            
            return 1/(tau*exp_norm_factor)*np.exp(-x/tau)

        def gauss(self, x, mu, sigma):
            
            return np.exp(-0.5*((x-mu)/sigma)**2)

        def norm_gauss(self, x, mu, sigma):
    
            norm_gauss_factor, _ = integrate.quad(gauss, self.xmass_min, self.xmass_max, args = [mu, sigma])
            norm_gauss = (1/norm_gauss_factor)*np.exp(-0.5*((x-mu)/sigma)**2)
    
            return norm_gauss

        def composite_pdf(self, x, mu, sigma, tau, f):
            """
            Composite PDF.
        
            Args:

            """
            norm_exp = norm_exp(x, tau)
            norm_gauss = norm_gauss(x, mu, sigma)
    
            return(f*norm_exp + (1-f)*norm_gauss)

        nll = UnbinnedNLL(self.xmass, composite_pdf)
        #set params to some vals
        minuit = Minuit(nll, x, mu, sigma, tau, f)
        minuit.limits["f"] = (0, 1)
        minuit.migrad()
        minuit.hesse()

        for i, j, k in zip(minuit.parameters, minuit.values, minuit.errors):
            print(f"{i} = {j} +/- {k}"")
        
        # Plotting
        x = np.linspace(1800, 2049, 1000)
        y = [composite_pdf(x, #vals of the params that we got) for i in x]

        plt.hist(xmass, bins = 100, color='orange')
        plt.plot(x, y)
        plt.title("Histogram of measurements from a mass distribution")
        plt.xlabel("Mass")
        plt.ylabel("Counts")
        plt.show()
     
                        
# Main entry point of the program
if __name__ == "__main__":
    

    f = open("kkp.bin","r")
    b = np.fromfile(f,dtype=np.float32)
    ncol = 7
    # number of events
    nevent = len(b)/ncol
    x = np.split(b, nevent)
    # make list of invariant mass of events
    xmass = []
    for i in range(0, len(xdata)):
        xmass.append(xdata[i][0])
        # read input arguments
    
    args = sys.argv
    
    if(len(args) != 2):
        
        print ("Usage python f2_project.py part_n")
        sys.exit()
        
    part_n = int(args[1]) 

                
    if part_n != 1:

        print("There are only 5 options. Please choose the part of the project you are interested by typing 1, 2, 3, 4 or 5.")
        sys.exit()
    
    f2_project = F2Project(xmass)
    
    if part_n == 1:
        
        f2_project.part_1()
    
    else:
        print("There are only 5 options. Please choose the part of the project you are interested by typing 1, 2, 3, 4 or 5.")
        sys.exit()
