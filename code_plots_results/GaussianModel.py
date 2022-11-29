# GaussianModel.py
# code developed by Katarina Zatkova, Una Alberti
# part of the DAH F2 Project: Make Accurate Measurements of Particle Masses

"""
This code contains the main scope of the project without the enhancements.
The methods defined here were developed to be used for analysing the LHCb dataset collected in 2011,
where a pair of oppositely charged kaons and a pion have been combined.
In this part of the project Gaussian PDFs are used for the fits.
"""


# Imports
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import csv


class GaussianModel(object):


    def __init__(self, xmass):
        
        self.xmass = np.array(xmass)
        # ranges of the D+ mass distribution
        self.xmass_min = 1.800
        self.xmass_max = 1.925

    def part_1(self, plotval=False):
        """
        Assignment: 
            1. Consider first the D+ peak which is the particle with the lowest mass, i.e. the left
            most peak in the plot. Construct a composite probability density function (PDF) for
            the invariant mass of the muon pairs, which contains two components:
            - A Gaussian shape to fit the D+ mass peak;
            - A shallow falling exponential to fit the background shape of the mass spectrum<br>
            underneath and around the peak.

        Output:
            We provide two plots as the outputs for this part to prove that we consider data from the first peak:
            - Histogram of the full scale mass distribution
            - Histogram showing the mass distribution for the D+ peak
            An array of selected data around the D+ mass peak region is returned.
        """
        
        # defining the PDFs
        def norm_exp(x, A):
            exp_norm_factor = np.exp(-self.xmass_min/A) - np.exp(-self.xmass_max/A)
            return ((1/(A*exp_norm_factor))*np.exp(-x/A))

        def gauss(x, mu, sigma):        
            return np.exp(-0.5*((x-mu)/sigma)**2)

        def norm_gauss(x, mu, sigma):
            return (1/(sigma*np.sqrt(2*np.pi)))*gauss(x, mu, sigma)

        def composite_pdf_part1(x, mu, sigma, A, f):
            return (f*norm_exp(x, A) + (1-f)*norm_gauss(x, mu, sigma))


        def plot_histogram(data, title, savetitle):

            plt.hist(data, bins = 100, color ='orange')
            plt.title(title)
            plt.xlabel("Mass [GeV]")
            plt.ylabel("Counts")
            plt.savefig("GaussianModel/Part1/" + savetitle + ".png")
            plt.show()

        xmass_dplus = self.xmass[(self.xmass <= self.xmass_max) & (self.xmass >= self.xmass_min)] 
        self.xmass_dplus = xmass_dplus

        if plotval == True:
            plot_histogram(data=self.xmass, title="Histogram of the full scale mass distribution", savetitle="fullscale_masshist")
            plot_histogram(data=self.xmass_dplus, title="Histogram showing the mass distribution for the D+ peak", savetitle="dpluspeak_masshist")

        return(norm_exp, norm_gauss, composite_pdf_part1)


    def part_2(self):
        """
        Assignment:
            2. Use this PDF in a Maximum Likelihood fit to determine the parameters of the PDF.
            Note that it is essential that the composite PDF remains normalised to 1 over the
            range of the fit.
            Determine the D+ meson mass and yield, and all other parameters, and their errors.
            You should be able to obtain the parameter errors directly from the minimization 
            engine of your choice. Depending on your choice you will be able to chose different
            methods. It would be good to show that you understand these by obtaining them yourself 
            from the parameters of the Gaussian signal fit — this is described in the data handling lectures.
            Plot the fitted signal shape on top of the data.
        
        Output:
            Prints out best fit parameter values with the statistical errors for the D+ mass + background distribution (in terminal and in a file).
            The fitted signal shape is plotted on top of the data.
        """

        def minimizer_part2(func):

            nll = UnbinnedNLL(self.xmass_dplus, func)
            minuit = Minuit(nll, mu = 1.870, sigma = 0.005, A = 0.9, f = 0.6)
            minuit.errordef = 0.5
            minuit.errors = (0.1, 0.001, 0.1, 0.1)
            minuit.limits["f"] = (0, 1)
            minuit.migrad()
            minuit.hesse()

            return (minuit.parameters, minuit.values, minuit.errors)

        def plot_histogram_withfit(data, xfit_data, yfit_data, title, savetitle):

            plt.hist(data, bins = 100, color='orange', label = "measurements", density = True)
            plt.plot(xfit_data, yfit_data, label = "fit")
            plt.title(str(title))
            plt.xlabel("Mass [GeV]")
            plt.ylabel("Counts")
            plt.legend()
            plt.savefig("GaussianModel/Part2/" + savetitle + ".png")
            plt.show()

        _, _, composite_pdf_part1 = self.part_1()
        params, vals, staterrors = minimizer_part2(func = composite_pdf_part1)

        # printing the best fit parameters with their statistical errors in a file
        with open('GaussianModel/Part2/fitparams.txt', 'w') as f:
            print("\n", file=f)
            print("The determined best fit parameters with the statistical errors from the Gaussian model fit are: \n", file=f)
            print("Parameter  =   Value     ±   Estat", file=f)
            print("μ" + "          =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]), file=f)
            print("σ" + "          =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]), file=f)
            print(str(params[2]) + "          =   " + "{0:1.3f}".format(vals[2]) + "     ±   " + "{0:1.3f}".format(staterrors[2]), file=f)
            print(str(params[3]) + "          =   " + "{0:1.4f}".format(vals[3]) + "    ±   " + "{0:1.4f}".format(staterrors[3]), file=f)
            print("\n", file=f)
            f.close()
        
        # printing the best fit parameters with their statistical errors in terminal window
        print("\n")
        print(f"The determined best fit parameters with the statistical errors from the Gaussian model fit are: \n")
        print("Parameter  =   Value     ±   Estat")
        print("μ" + "          =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]))
        print("σ" + "          =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]))
        print(str(params[2]) + "          =   " + "{0:1.3f}".format(vals[2]) + "     ±   " + "{0:1.3f}".format(staterrors[2]))
        print(str(params[3]) + "          =   " + "{0:1.4f}".format(vals[3]) + "    ±   " + "{0:1.4f}".format(staterrors[3]))
        print("\n")
        
        #calculating the yield
        counts, edges, _ = plt.hist(self.xmass_dplus, bins = 100, color='orange')
        plt.close()
        for i in range(len(edges)):
            if (edges[i] > vals[0]) and (edges[i-1] < vals[0]):
                yield_val = counts[i-1]
                yield_err = ((counts[i-2]/2) + (counts[i]/2))/2
                print("The signal yield for the D+ meson is " + str(int(yield_val)) + " ± " + str(int(yield_err)) +".")
                
        xfit = np.linspace(self.xmass_min, self.xmass_max, 10000)
        yfit = composite_pdf_part1(xfit, mu = vals["mu"], sigma = vals["sigma"], A = vals["A"], f = vals["f"])

        # plotting histogram with the fit 
        plot_histogram_withfit(self.xmass_dplus, xfit, yfit, "Histogram of the mass distribution for the D+ peak with the fitted signal", "dpluspeak_masshist_withfit")


    def part_3(self, plotprintval = False):
        """
        Assignment:
            3. Now consider the entire mass range, and perform a simultaneous fit for both peaks,
            and the underlying background. Again you should always report the parameter values,
            and their errors. Plot the fitted signal shape on top of the data.

        Outputs:
            Prints out best fit parameter values with the statistical errors for the entire mass range (in terminal and in a file).
            The fitted signal shape is plotted on top of the data.
        """
        self.xmass_min = 1.80
        self.xmass_max = 2.05

        def composite_pdf_part3(x, mu_1, sigma_1, mu_2, sigma_2, A, f_1, f_2):
            norm_exp, norm_gauss, _ = self.part_1()
            return ((f_1)*norm_gauss(x, mu_1, sigma_1) + (f_2)*norm_gauss(x, mu_2, sigma_2) + (1-(f_1 + f_2))*norm_exp(x, A))
        

        def minimizer_part3(func):

            nll = UnbinnedNLL(self.xmass, func)
            minuit = Minuit(nll, mu_1 = 1.87, sigma_1 = 0.005, mu_2 = 1.97, sigma_2 = 0.005, A = 0.5, f_1 = 0.15, f_2 = 0.5)
            minuit.errordef = 0.5
            minuit.errors = [0.1, 0.001, 0.1, 0.001,0.1,0.1,0.1]
            minuit.limits["f_1"] = (0, 1)
            minuit.limits["f_2"] = (0, 1)
            minuit.migrad()
            minuit.hesse()

            return (minuit.parameters, minuit.values, minuit.errors)


        def plot_histogram_withfit(data, xfit_data, yfit_data, title, savetitle):

            plt.hist(data, bins = 500, color='orange', label = "measurements", density = True)
            plt.plot(xfit_data, yfit_data, label = "fit")
            plt.title(str(title))
            plt.xlabel("Mass [GeV]")
            plt.ylabel("Counts")
            plt.legend()
            plt.savefig("GaussianModel/Part3/" + savetitle + ".png")
            plt.show()

        params, vals, staterrors = minimizer_part3(func = composite_pdf_part3)

        if plotprintval == True:
            # printing the best fit parameters with their statistical errors in a file
            with open('GaussianModel/Part3/fitparams.txt', 'w') as f:
                print("\n", file=f)
                print("The determined best fit parameters with the statistical errors from the Gaussian model fit are: \n", file=f)
                print("Parameter  =   Value     ±   Estat", file=f)
                print("μ1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]), file=f)
                print("σ1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]), file=f)
                print("μ2" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ±   " + "{0:1.6f}".format(staterrors[2]), file=f)
                print("σ2" + "         =   " + "{0:1.7f}".format(vals[3]) + " ±   " + "{0:1.7f}".format(staterrors[3]), file=f)
                print(str(params[4]) + "          =   " + "{0:1.4f}".format(vals[4]) + "    ±   " + "{0:1.4f}".format(staterrors[4]), file=f)
                print("f1" + "         =   " + "{0:1.5f}".format(vals[5]) + "   ±   " + "{0:1.5f}".format(staterrors[5]), file=f)
                print("f2" + "         =   " + "{0:1.5f}".format(vals[6]) + "   ±   " + "{0:1.5f}".format(staterrors[6]), file=f)
                print("\n", file=f)
                f.close()

            # printing the best fit parameters with their statistical errors in terminal window
            print("\n")
            print(f"The determined best fit parameters with the statistical errors from the Gaussian model fit are: \n")
            print("Parameter  =   Value     ±   Estat")
            print("μ1" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]))
            print("σ1" + "         =   " + "{0:1.6f}".format(vals[1]) + "  ±   " + "{0:1.6f}".format(staterrors[1]))
            print("μ2" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ±   " + "{0:1.6f}".format(staterrors[2]))
            print("σ2" + "         =   " + "{0:1.7f}".format(vals[3]) + " ±   " + "{0:1.7f}".format(staterrors[3]))
            print(str(params[4]) + "          =   " + "{0:1.4f}".format(vals[4]) + "    ±   " + "{0:1.4f}".format(staterrors[4]))
            print("f1" + "         =   " + "{0:1.5f}".format(vals[5]) + "   ±   " + "{0:1.5f}".format(staterrors[5]))
            print("f2" + "         =   " + "{0:1.5f}".format(vals[6]) + "   ±   " + "{0:1.5f}".format(staterrors[6]))
            print("\n")

            """
            with open('mu1_allmodels.csv', 'w') as f1:
                mu1_val = vals[0]
                mu1_err = staterrors[0]
                writer = csv.writer(f1, delimiter='\t')
                writer.writerow([mu1_val, mu1_err, "Gaussian Model"])
                f1.close()
            
            with open('mu2_allmodels.csv', 'w') as f2:
                mu2_val = vals[2]
                mu2_err = staterrors[2]
                writer = csv.writer(f2, delimiter='\t')
                writer.writerow([mu2_val, mu2_err, "Gaussian Model"])
                f2.close()
            """
            
            print("Peak 1: μ" + "         =   " + "{0:1.6f}".format(vals[0]) + "  ±   " + "{0:1.6f}".format(staterrors[0]))
            print("Peak 2: μ" + "         =   " + "{0:1.6f}".format(vals[2]) + "  ±   " + "{0:1.6f}".format(staterrors[2]))
            print("\n")
            
            xfit = np.linspace(self.xmass_min, self.xmass_max, 10000)
            yfit = composite_pdf_part3(xfit, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], sigma_2 = vals["sigma_2"], A = vals["A"], f_1 = vals["f_1"], f_2 = vals["f_2"])

            # plotting histogram with the fit 
            plot_histogram_withfit(self.xmass, xfit, yfit, "Histogram of the full scale mass distribution with the fitted signal", "fullscale_masshist_withfit")
            
        else:
            return (composite_pdf_part3, vals)
    
    def part_4(self):
        """
        Assignment:
            4. The results so far probably look quite good by eye, i.e. the signal shape plotted on
            top of the data probably looks like it fits well. However this can be misleading when
            performing a precision measurement. You should make a plot of what are called the
            “residuals.” A residual is the difference between the data in the binned histogram and
            the best-fit mass model value for the centre of that bin. Describe what you see.
        Output:
            Plot of the residuals.
        """

        composite_pdf_part3, vals = self.part_3()
        halfofbinsize = 0.00025
        
        counts = plt.hist(self.xmass, bins = 500, color='orange', density=True)
        plt.close()

        x = np.linspace(self.xmass_min + halfofbinsize,self.xmass_max - halfofbinsize, 500)
        y = composite_pdf_part3(x, mu_1 = vals["mu_1"], sigma_1 = vals["sigma_1"], mu_2 = vals["mu_2"], sigma_2 = vals["sigma_2"], A = vals["A"], f_1 = vals["f_1"], f_2 = vals["f_2"])

        residuals = np.subtract(counts[0],y)

        plt.plot(x,residuals)
        plt.title("Residuals from the Gaussian Model")
        plt.xlabel("Mass [GeV]")
        plt.ylabel("Residuals")
        plt.savefig("GaussianModel/Part4/residuals.png")
        plt.show()
