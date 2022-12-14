# SumOfGaussiansModel.py
# code developed by Katarina Zatkova, Una Alberti
# part of the DAH F2 Project: Make Accurate Measurements of Particle Masses

"""
This code contains the enhancements beyond the scope of the project.
The methods defined here were developed to be used for analysing the LHCb dataset collected in 2011,
where a pair of oppositely charged kaons and a pion have been combined.
In this part of the project PDF comprising a function which is the sum of two
Gaussian functions (i.e. one narrow and one wide Gaussian function to fit a sin-
gle D meson peak) is used for the fit.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import csv


class SumOfGaussiansModel(object):


    def __init__(self, xmass):
        
        self.xmass = np.array(xmass)
        # ranges of the mass distribution
        self.xmass_min = 1.800
        self.xmass_max = 2.05

    def norm_2gauss_summed(self, x, mu_1, sigma_1, mu_2, sigma_2):
        norm_factor = (sigma_1 + sigma_2)*np.sqrt(2*np.pi)
        norm_gauss = (1/(norm_factor))*(np.exp(-0.5*(((x-mu_1)/sigma_1)**2)) + np.exp(-0.5*(((x-mu_2)/sigma_2)**2)))
        return norm_gauss

    def norm_exp(self, x, A):
        exp_norm_factor = np.exp(-self.xmass_min/A) - np.exp(-self.xmass_max/A)
        return ((1/(A*exp_norm_factor))*np.exp(-x/A))

    def composite_pdf(self, x, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, A, f_1, f_2):
        return ((f_1)*self.norm_2gauss_summed(x, mu_1, sigma_1, mu_2, sigma_2) + (f_2)*self.norm_2gauss_summed(x, mu_3, sigma_3, mu_4, sigma_4) + (1-(f_1+f_2))*self.norm_exp(x, A))

    def minimizer(self, func):
        nll = UnbinnedNLL(self.xmass, func)
        minuit = Minuit(nll,mu_1=1.87, sigma_1=0.004, mu_2=1.87, sigma_2=0.007, mu_3 = 1.97, sigma_3=0.004, 
                        mu_4=1.97,  sigma_4=0.007, A=0.5, f_1=0.15, f_2=0.5)
        minuit.errordef = 0.5
        minuit.errors = [0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1,0.1,0.1]
        minuit.limits["f_1"] = (0, 1)
        minuit.limits["f_2"] = (0, 1)
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
        plt.savefig("SumOfGaussiansModel/" + savetitle + ".png")
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
            with open('SumOfGaussiansModel/fitparams.txt', 'w') as f:
                print("\n", file=f)
                print("The determined best fit parameters with the statistical errors from the Sum-Of-Gaussians model fit are: \n", file=f)
                print("Parameter  =   Value     ??   Estat", file=f)
                print("??1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ??   " + "{0:1.6f}".format(staterrors[0]), file=f)
                print("??1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ??   " + "{0:1.6f}".format(staterrors[1]), file=f)
                print("??2" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ??   " + "{0:1.6f}".format(staterrors[2]), file=f)
                print("??2" + "         =   " + "{0:1.6f}".format(vals[3]) + "  ??   " + "{0:1.6f}".format(staterrors[3]), file=f)
                print("??3" + "         =   " + "{0:1.6f}".format(vals[4]) + "  ??   " + "{0:1.6f}".format(staterrors[4]), file=f)
                print("??3" + "         =   " + "{0:1.6f}".format(vals[5]) + "  ??   " + "{0:1.6f}".format(staterrors[5]), file=f)
                print("??4" + "         =   " + "{0:1.6f}".format(vals[6]) + "  ??   " + "{0:1.6f}".format(staterrors[6]), file=f)
                print("??4" + "         =   " + "{0:1.6f}".format(vals[7]) + "  ??   " + "{0:1.6f}".format(staterrors[7]), file=f)
                print(str(params[8]) + "          =   " + "{0:1.4f}".format(vals[8]) + "    ??   " + "{0:1.4f}".format(staterrors[8]), file=f)
                print("f1" + "         =   " + "{0:1.5f}".format(vals[9]) + "   ??   " + "{0:1.5f}".format(staterrors[9]), file=f)
                print("f2" + "         =   " + "{0:1.5f}".format(vals[10]) + "   ??   " + "{0:1.5f}".format(staterrors[10]), file=f)
                print("\n", file=f)
                f.close()
            
            # printing the best fit parameters with their statistical errors in terminal window
            print("\n")
            print(f"The determined best fit parameters with the statistical errors from the Sum-Of-Gaussians model fit are: \n")
            print("Parameter  =   Value     ??   Estat")
            print("??1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ??   " + "{0:1.6f}".format(staterrors[0]))
            print("??1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ??   " + "{0:1.6f}".format(staterrors[1]))
            print("??2" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ??   " + "{0:1.6f}".format(staterrors[2]))
            print("??2" + "         =   " + "{0:1.6f}".format(vals[3]) + "  ??   " + "{0:1.6f}".format(staterrors[3]))
            print("??3" + "         =   " + "{0:1.6f}".format(vals[4]) + "  ??   " + "{0:1.6f}".format(staterrors[4]))
            print("??3" + "         =   " + "{0:1.6f}".format(vals[5]) + "  ??   " + "{0:1.6f}".format(staterrors[5]))
            print("??4" + "         =   " + "{0:1.6f}".format(vals[6]) + "  ??   " + "{0:1.6f}".format(staterrors[6]))
            print("??4" + "         =   " + "{0:1.6f}".format(vals[7]) + "  ??   " + "{0:1.6f}".format(staterrors[7]))
            print(str(params[8]) + "          =   " + "{0:1.4f}".format(vals[8]) + "    ??   " + "{0:1.4f}".format(staterrors[8]))
            print("f1" + "         =   " + "{0:1.5f}".format(vals[9]) + "   ??   " + "{0:1.5f}".format(staterrors[9]))
            print("f2" + "         =   " + "{0:1.5f}".format(vals[10]) + "   ??   " + "{0:1.5f}".format(staterrors[10]))
            print("\n")

            mu_peak1_av = np.mean([vals[0], vals[2]])
            mu_peak2_av = np.mean([vals[4], vals[6]])
            mu_peak1_err = np.sqrt(staterrors[0]**2 + staterrors[2]**2)/2
            mu_peak2_err = np.sqrt(staterrors[4]**2 + staterrors[6]**2)/2

            """
            with open('mu1_allmodels.csv', 'a') as f1:
                writer = csv.writer(f1, delimiter='\t')
                writer.writerow([mu_peak1_av, mu_peak1_err, "Sum-Of-Gaussians Model"])
                f1.close()
            
            with open('mu2_allmodels.csv', 'a') as f2:
                writer = csv.writer(f2, delimiter='\t')
                writer.writerow([mu_peak2_av, mu_peak2_err, "Sum-Of-Gaussians Model"])
                f2.close()
            """
            
            print("Peak 1: ??" + "         =   " + "{0:1.6f}".format(mu_peak1_av) + "  ??   " + "{0:1.6f}".format(mu_peak1_err))
            print("Peak 2: ??" + "         =   " + "{0:1.6f}".format(mu_peak2_av) + "  ??   " + "{0:1.6f}".format(mu_peak2_err))
            print("\n")
            
            xfit = np.linspace(self.xmass_min, self.xmass_max, 10000)
            yfit = self.composite_pdf(xfit, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], 
                                    sigma_2 = vals["sigma_2"], mu_3 = vals["mu_3"], sigma_3 = vals["sigma_3"],
                                    mu_4 = vals["mu_4"], sigma_4 = vals["sigma_4"], A = vals["A"], 
                                    f_1 = vals["f_1"],f_2 = vals["f_2"])

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
        y = self.composite_pdf(x, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], 
                                    sigma_2 = vals["sigma_2"], mu_3 = vals["mu_3"], sigma_3 = vals["sigma_3"],
                                    mu_4 = vals["mu_4"], sigma_4 = vals["sigma_4"], A = vals["A"], 
                                    f_1 = vals["f_1"],f_2 = vals["f_2"])

        residuals = np.subtract(counts[0],y)
        
        plt.plot(x,residuals)
        plt.title("Residuals from the Sum-Of-Gaussians Model")
        plt.xlabel("Mass [GeV]")
        plt.ylabel("Residuals")
        plt.savefig("SumOfGaussiansModel/residuals.png")
        plt.show()
