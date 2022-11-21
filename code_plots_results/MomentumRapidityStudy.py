# MomentumRapidityStudy.py
# code developed by Katarina Zatkova, Una Alberti
# part of the DAH F2 Project: Make Accurate Measurements of Particle Masses

"""
This code contains the enhancements beyond the scope of the project.
The methods defined here were developed to be used for analysing the LHCb dataset collected in 2011,
where a pair of oppositely charged kaons and a pion have been combined.
In this part of the project, a study on how the mass and the width of the peak (the resolution) depends 
on the transverse momentum (p⊥) and rapidity (η) was conducted. 
The model with a Gaussian PDF function was chosen for this study.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import crystalball
import scipy.integrate as integrate


class MomentumRapidityStudy(object):


    def __init__(self, xmass, momentum, rapidity):
        
        self.xmass = np.array(xmass)
        self.momentum = np.array(momentum)
        self.rapidity = np.array(rapidity)
        self.xmass_min = 1.80
        self.xmass_max = 2.05
    
    def norm_exp(self, x, A):
        exp_norm_factor = np.exp(-self.xmass_min/A) - np.exp(-self.xmass_max/A)
        return ((1/(A*exp_norm_factor))*np.exp(-x/A))

    def gauss(self, x, mu, sigma):        
        return np.exp(-0.5*((x-mu)/sigma)**2)

    def norm_gauss(self, x, mu, sigma):
        return (1/(sigma*np.sqrt(2*np.pi)))*self.gauss(x, mu, sigma)
    
    def composite_pdf(self, x, mu_1, sigma_1, mu_2, sigma_2, A, f_1, f_2):
        return ((f_1)*self.norm_gauss(x, mu_1, sigma_1) + (f_2)*self.norm_gauss(x, mu_2, sigma_2) + (1-(f_1 + f_2))*self.norm_exp(x, A))

    def minimizer(self, mass_data, func):
        nll = UnbinnedNLL(mass_data, func)
        minuit = Minuit(nll, mu_1 = 1.87, sigma_1 = 0.005, mu_2 = 1.97, sigma_2 = 0.005, A = 0.5, f_1 = 0.15, f_2 = 0.5)
        minuit.errordef = 0.5
        minuit.errors = [0.1, 0.001, 0.1, 0.001,0.1,0.1,0.1]
        minuit.limits["f_1"] = (0, 1)
        minuit.limits["f_2"] = (0, 1)
        minuit.migrad()
        minuit.hesse()
        vals = minuit.values
        staterrors = minuit.errors

        mu_peak1 = vals[0]
        mu_peak2 = vals[2]
        mu_peak1_errstat = staterrors[0]
        mu_peak2_errstat = staterrors[2]
        sigma_peak1 = vals[1]
        sigma_peak2 = vals[3]
        sigma_peak1_errstat = staterrors[1]
        sigma_peak2_errstat = staterrors[3]

        mu_peak1_errsys = 0.0000038
        mu_peak2_errsys = 0.000046
        sigma_peak1_errsys = 0.000635
        sigma_peak2_errsys = 0.000746

        mu_peak1_err = np.sqrt(mu_peak1_errstat**2 + mu_peak1_errsys**2)
        mu_peak2_err = np.sqrt(mu_peak2_errstat**2 + mu_peak2_errsys**2)
        sigma_peak1_err = np.sqrt(sigma_peak1_errstat**2 + sigma_peak1_errsys**2)
        sigma_peak2_err = np.sqrt(sigma_peak2_errstat**2 + sigma_peak2_errsys**2)

        return (mu_peak1, mu_peak2, mu_peak1_err, mu_peak2_err, sigma_peak1, sigma_peak2, sigma_peak1_err, sigma_peak2_err)


    def plot_mass_vs_momentum(self):
        
        momentum_ind_1, momentum_ind_2, momentum_ind_3, momentum_ind_4 = [], [], [], []

        for i in range(0, len(self.xmass)):
            
            if self.momentum[i] < (11.7):
                momentum_ind_1.append(i)  
            elif self.momentum[i] > (11.7) and  self.momentum[i] < (22.6):
                momentum_ind_2.append(i)
            elif self.momentum[i] > (22.6) and self.momentum[i] < (33.5):
                momentum_ind_3.append(i)    
            elif self.momentum[i] > (33.5):
                momentum_ind_4.append(i)  
            else:
                pass

        mass_1, mass_2, mass_3, mass_4  = [], [], [], []        
        
        for index in momentum_ind_1:
            mass_1.append(self.xmass[index])      
        for index in momentum_ind_2:
            mass_2.append(self.xmass[index])     
        for index in momentum_ind_3:
            mass_3.append(self.xmass[index])      
        for index in momentum_ind_4:
            mass_4.append(self.xmass[index])

        #now we do a seperate fit for all 4 regions
        # region 1   
        mu1_1, mu2_1, mu1_err1, mu2_err1, sigma1_1, sigma2_1, sigma1_err1, sigma2_err1 = self.minimizer(mass_1, self.composite_pdf)
        # region 2   
        mu1_2, mu2_2, mu1_err2, mu2_err2, sigma1_2, sigma2_2, sigma1_err2, sigma2_err2 = self.minimizer(mass_2, self.composite_pdf)
        # region 3   
        mu1_3, mu2_3, mu1_err3, mu2_err3, sigma1_3, sigma2_3, sigma1_err3, sigma2_err3 = self.minimizer(mass_3, self.composite_pdf)
        # region 4
        mu1_4, mu2_4, mu1_err4, mu2_err4, sigma1_4, sigma2_4, sigma1_err4, sigma2_err4 = self.minimizer(mass_4, self.composite_pdf)

        #make plot of region (1,2,3) vs mass
        region_n = [1,2,3]

        m1 = [mu1_1, mu1_2, mu1_3]
        m2 = [mu2_1, mu2_2, mu2_3]
        m1_err = [mu1_err1, mu1_err2, mu1_err3]
        m2_err = [mu2_err1, mu2_err2, mu2_err3]
        
        s1 = [sigma1_1, sigma1_2, sigma1_3]
        s2 = [sigma2_1, sigma2_2, sigma2_3]
        s1_err = [sigma1_err1, sigma1_err2, sigma1_err3]
        s2_err = [sigma2_err1, sigma2_err2, sigma2_err3]

        plt.errorbar(region_n, m1, yerr = m1_err, fmt="o", barsabove=True, capsize=10)
        plt.title('Masses of D+(s) peak in 3 (p⊥) Momentum Regions ')
        plt.ylabel('Mass [Gev]')
        plt.xlabel('p⊥ Region')
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_D+mass_vs_momentum.png")
        plt.show()

        plt.errorbar(region_n, m2, yerr = m2_err, fmt="o", barsabove=True, capsize=10)
        plt.title('Masses of D0 peak in 3 (p⊥) Momentum Regions ')
        plt.ylabel('Mass [Gev]')
        plt.xlabel('p⊥ Region')
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_D0mass_vs_momentum.png")
        plt.show()

        plt.errorbar(region_n, s1, yerr=s1_err, fmt="o", barsabove=True,label='D+(s)',markersize = 4, capsize=10)
        plt.errorbar(region_n, s2, yerr=s2_err, fmt="o", barsabove=True,label='D0',markersize = 4, capsize=10)
        plt.title('Spread of D+(s) and D0 peaks for 3 (p⊥) Momentum Regions')
        plt.ylabel('Sigma [Gev]')
        plt.xlabel('p⊥ Region')
        plt.legend()
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_sigma_vs_momentum.png")
        plt.show()


    def plot_mass_vs_rapidity(self):
        
        rapidity_ind_1, rapidity_ind_2, rapidity_ind_3, rapidity_ind_4 = [], [], [], []

        for i in range(0, len(self.xmass)):
            
            if self.rapidity[i] < (2.675):
                rapidity_ind_1.append(i)
            elif self.rapidity[i] > (2.675) and  self.rapidity[i] < (3.55):
                rapidity_ind_2.append(i)
            elif self.rapidity[i] > (3.55) and self.rapidity[i] < (4.425):
                rapidity_ind_3.append(i)  
            elif self.rapidity[i] > (4.425):
                rapidity_ind_4.append(i)
            else:
                pass


        mass_1, mass_2, mass_3, mass_4  = [], [], [], []        
        
        for index in rapidity_ind_1:
            mass_1.append(self.xmass[index])      
        for index in rapidity_ind_2:
            mass_2.append(self.xmass[index])     
        for index in rapidity_ind_3:
            mass_3.append(self.xmass[index])      
        for index in rapidity_ind_4:
            mass_4.append(self.xmass[index])

        #now we do a seperate fit for all 4 regions
        # region 1   
        mu1_1, mu2_1, mu1_err1, mu2_err1, sigma1_1, sigma2_1, sigma1_err1, sigma2_err1 = self.minimizer(mass_1, self.composite_pdf)
        # region 2   
        mu1_2, mu2_2, mu1_err2, mu2_err2, sigma1_2, sigma2_2, sigma1_err2, sigma2_err2 = self.minimizer(mass_2, self.composite_pdf)
        # region 3   
        mu1_3, mu2_3, mu1_err3, mu2_err3, sigma1_3, sigma2_3, sigma1_err3, sigma2_err3 = self.minimizer(mass_3, self.composite_pdf)
        # region 4
        mu1_4, mu2_4, mu1_err4, mu2_err4, sigma1_4, sigma2_4, sigma1_err4, sigma2_err4 = self.minimizer(mass_4, self.composite_pdf)

        #make plot of region (1,2,3) vs mass
        region_n = [1,2,3,4]

        m1 = [mu1_1, mu1_2, mu1_3, mu1_4]
        m2 = [mu2_1, mu2_2, mu2_3, mu2_4]
        m1_err = [mu1_err1, mu1_err2, mu1_err3, mu1_err4]
        m2_err = [mu2_err1, mu2_err2, mu2_err3, mu2_err4]
        
        s1 = [sigma1_1, sigma1_2, sigma1_3, sigma1_4]
        s2 = [sigma2_1, sigma2_2, sigma2_3, sigma2_4]
        s1_err = [sigma1_err1, sigma1_err2, sigma1_err3, sigma1_err4]
        s2_err = [sigma2_err1, sigma2_err2, sigma2_err3, sigma2_err4]

        plt.errorbar(region_n, m1, yerr = m1_err, fmt="o", barsabove=True, capsize=10)
        plt.title('Masses of D+ (s) peak in 4 Rapidity (η) Regions')
        plt.ylabel('Mass [Gev]')
        plt.xlabel('η Region')
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_D+mass_vs_rapidity.png")
        plt.show()

        plt.errorbar(region_n, m2, yerr = m2_err, fmt="o", barsabove=True, capsize=10)
        plt.title('Masses of D0(s) peak in 4 Rapidity (η) Regions')
        plt.ylabel('Mass [Gev]')
        plt.xlabel('η Region')
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_D0mass_vs_rapidity.png")
        plt.show()

        plt.errorbar(region_n, s1, yerr=s1_err, fmt="o", barsabove=True,label='D+(s)',markersize = 4, capsize=10)
        plt.errorbar(region_n, s2, yerr=s2_err, fmt="o", barsabove=True,label='D0',markersize = 4, capsize=10)
        plt.title('Spread of D+ (s) and D0 peaks for 4 Rapidity (η) Regions')
        plt.ylabel('Mass [Gev]')
        plt.xlabel('η Region')
        plt.legend()
        plt.grid(True)
        plt.savefig("MomentumRapidityStudy/plot_sigma_vs_rapidity.png")
        plt.show()
