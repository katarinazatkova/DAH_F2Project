# CrystalBallModel.py
# code developed by Katarina Zatkova, Una Alberti
# part of the DAH F2 Project: Make Accurate Measurements of Particle Masses

"""
This code contains the enhancements beyond the scope of the project.
The methods defined here were developed to be used for analysing the LHCb dataset collected in 2011,
where a pair of oppositely charged kaons and a pion have been combined.
In this part of the project Crystal Ball PDF is used for the fit.
Crystal Ball function incorporates a non-Gaussian tail at the lower end of the mass peak.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import crystalball
import csv


class CrystalBallModel(object):


    def __init__(self, xmass):
        
        self.xmass = np.array(xmass)
        # ranges of the mass distribution
        self.xmass_min = 1.800
        self.xmass_max = 2.05
        self.beta_1 = 2
        self.beta_2 = 2
        self.m_2 = 1.1
        self.m_1 = 1.1

    def norm_exp(self, x, A):
        exp_norm_factor = np.exp(-self.xmass_min/A) - np.exp(-self.xmass_max/A)
        return ((1/(A*exp_norm_factor))*np.exp(-x/A))

    def crystal_ball(self, x, beta, m, mu , sigma):
        return crystalball.pdf(x, beta, m, mu , sigma)

    def composite_pdf(self, x, mu_1 , sigma_1, mu_2 , sigma_2, A, f_1, f_2):
        return ((f_1)*self.crystal_ball(x, self.beta_1, self.m_1, mu_1, sigma_1) + (f_2)*self.crystal_ball(x, self.beta_2, self.m_2, mu_2, sigma_2) + (1-(f_1+f_2))*self.norm_exp(x, A))

    def minimizer(self, func):
        nll = UnbinnedNLL(self.xmass, func)
        minuit = Minuit(nll, mu_1 = 1.87, sigma_1=0.005, mu_2=1.97 , sigma_2=0.006, A=0.5, f_1=0.15, f_2=0.5)
        minuit.errordef = 0.5
        minuit.errors = [0.001, 0.001, 0.001, 0.001,0.001,0.001,0.001]
        minuit.limits["f_1"] = (0, 1)
        minuit.limits["f_2"] = (0, 1)
        minuit.limits["mu_1"] = (1.84, 1.9)
        minuit.limits["mu_2"] = (1.94, 2)
        minuit.migrad()
        minuit.hesse()
        return (minuit.parameters, minuit.values, minuit.errors)
    
    def plot_histogram_withfit(self, data, xfit_data, yfit_data, title, savetitle):

        plt.hist(data, bins = 500, color='orange', label = "measurements", density = True)
        plt.plot(xfit_data, yfit_data, label = "fit")
        plt.title(str(title))
        plt.xlabel("Mass [GeV]")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig("CrystalBallModel/" + savetitle + ".png")
        plt.show()
    
    def fit(self, plotprintval = False):
        """
        
        Outputs:
            Prints out best fit parameter values with the statistical errors for the entire mass range (in terminal and in a file).
            The fitted signal shape is plotted on top of the data.
        """

        params, vals, staterrors = self.minimizer(func = self.composite_pdf)

        if plotprintval == True:
            # printing the best fit parameters with their statistical errors in a file
            with open('CrystalBallModel/fitparams.txt', 'w') as f:
                print("\n", file=f)
                print("The determined best fit parameters with the statistical errors from the Crystal Ball model fit are: \n", file=f)
                print("Parameter  =   Value     ±   Estat", file=f)
                print("μ1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]), file=f)
                print("σ1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]), file=f)
                print("μ2" + "         =   " + "{0:1.7f}".format(vals[2]) + " ±   " + "{0:1.7f}".format(staterrors[2]), file=f)
                print("σ2" + "         =   " + "{0:1.7f}".format(vals[3]) + " ±   " + "{0:1.7f}".format(staterrors[3]), file=f)
                print(str(params[4]) + "          =   " + "{0:1.3f}".format(vals[4]) + "     ±   " + "{0:1.3f}".format(staterrors[4]), file=f)
                print("f1" + "         =   " + "{0:1.5f}".format(vals[5]) + "   ±   " + "{0:1.5f}".format(staterrors[5]), file=f)
                print("f2" + "         =   " + "{0:1.5f}".format(vals[6]) + "   ±   " + "{0:1.5f}".format(staterrors[6]), file=f)
                print("\n", file=f)
                f.close()
            
            # printing the best fit parameters with their statistical errors in terminal window
            print("\n")
            print(f"The determined best fit parameters with the statistical errors from the Crystal Ball model fit are: \n")
            print("Parameter  =   Value     ±   Estat")
            print("μ1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]))
            print("σ1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]))
            print("μ2" + "         =   " + "{0:1.7f}".format(vals[2]) + " ±   " + "{0:1.7f}".format(staterrors[2]))
            print("σ2" + "         =   " + "{0:1.7f}".format(vals[3]) + " ±   " + "{0:1.7f}".format(staterrors[3]))
            print(str(params[4]) + "          =   " + "{0:1.3f}".format(vals[4]) + "     ±   " + "{0:1.3f}".format(staterrors[4]))
            print("f1" + "         =   " + "{0:1.5f}".format(vals[5]) + "   ±   " + "{0:1.5f}".format(staterrors[5]))
            print("f2" + "         =   " + "{0:1.5f}".format(vals[6]) + "   ±   " + "{0:1.5f}".format(staterrors[6]))
            print("\n")

            """
            with open('mu1_allmodels.csv', 'a') as f1:
                mu1_val = vals[0]
                mu1_err = staterrors[0]
                writer = csv.writer(f1, delimiter='\t')
                writer.writerow([mu1_val, mu1_err, "Crystal Ball Model"])
                f1.close()
            
            with open('mu2_allmodels.csv', 'a') as f2:
                mu2_val = vals[2]
                mu2_err = staterrors[2]
                writer = csv.writer(f2, delimiter='\t')
                writer.writerow([mu2_val, mu2_err, "Crystal Ball Model"])
                f2.close()
            """
            
            print("Peak 1: μ" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]))
            print("Peak 2: μ" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ±   " + "{0:1.6f}".format(staterrors[2]))
            print("\n")
            
            xfit = np.linspace(self.xmass_min, self.xmass_max, 10000)
            yfit = self.composite_pdf(xfit, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], sigma_2 = vals["sigma_2"], A = vals["A"], f_1 = vals["f_1"], f_2 = vals["f_2"])

            # plotting histogram with the fit 
            self.plot_histogram_withfit(self.xmass, xfit, yfit, "Histogram of the full scale mass distribution with the fitted signal", "fullscale_masshist_withfit")
        else:
            return vals
    
    def get_residuals(self):
        """
        Output:
            Plot of the residuals.
        """
        vals = self.fit()
        halfofbinsize = 0.00025

        counts = plt.hist(self.xmass, bins = 500, color='orange', density=True)
        plt.close()

        x = np.linspace(self.xmass_min + halfofbinsize,self.xmass_max - halfofbinsize, 500)
        y = self.composite_pdf(x, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], sigma_2 = vals["sigma_2"], A = vals["A"], f_1 = vals["f_1"], f_2 = vals["f_2"])

        residuals = np.subtract(counts[0],y)
        
        plt.plot(x,residuals)
        plt.title("Residuals from the Crystal Ball Model")
        plt.xlabel("Mass [GeV]")
        plt.ylabel("Residuals")
        plt.savefig("CrystalBallModel/residuals.png")
        plt.show()
