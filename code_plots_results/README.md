
Authors: Katarina Zatkova, Una Alberti

Date: November 21, 2022

This code was developed for Data Acquisition and Handling course F2 Project: Make Accurate Measurements of Particle Masses. It uses LHCb dataset collected in 2011 where a pair of oppositely charged kaons and a pions have been combined. Two clear peaks are observed in this mass spectrum corresponding to the D+s (quark content cs) and D+ (quark content cd) mesons (charge conjugation is implied). This data was analysed using the maximum likelihood process to fit different mass model shapes to the data. From this the parameters of the mass model for the signal peaks, and their errors were determined.

The mass models that were implemented for this project are:

    Gaussian Model - Gaussian PDFs were used for the fits of the two peaks.

    Sum of Gaussians Model - a PDF comprising a function which is the sum of two Gaussian functions (i.e. one narrow and one wide Gaussian function to fit a single D meson peak) is used for the fit for both peaks.

    Crystal Ball Model - Crystal Ball PDF is used for the fit. Crystal Ball function incorporates a non-Gaussian tail at the lower end of the mass peak.
    
    Sum of Crystal Ball and Gaussian -  a PDF comprising a function which is the sum of a Gaussian function and a Crystal ball function (i.e. one Gaussian and one Crystall ball function to fit a single D meson peak) is used for the fit for each peak.

Systematic errors due to the model were also computed using the shift method.

This code also performs a study of how the mass and the width of the peak (the resolution) depends on the transverse momentum (p⊥) and rapidity (η).

The environment that was used:

channels:
  - conda-forge

dependencies:
  - python=3.9
  - matplotlib
  - numpy
  - geojson
  - pandas
  - pydotplus
  - scipy
  - jupyter
  - ipython
  - ipykernel
  - iminuit
  - folium
  - tensorflow
  - scikit-learn
  - seaborn

Usage: 'python3 main.py Model Part_number'
(see information below)

Files structure:

    main.py  - the core program that uses GaussianModel, SumOfGaussiansModel, CrystalBallModel, SumOfCrystalBallGaussianModel and MomentumRapidityStudy     classes and is used to perform the analysis.

            It prompts the user to input:   Model(str) - There are models to choose from:   'GaussianModel'
                                                                `                           'SumOfGaussiansModel' 
                                                                                            'CrystalBallModel' 
                                                                                            'SumOfCrystalBallGaussianModel' 
                                                                                            'Other' -  choose 'Other' to see the results for the momentum and rapidity study
                                            
                                            Part_number(int) -  This parameter is only applicable for the 'GaussianModel'.
                                                                Choose the results from which part (question) of the main project assignment one wishes to see (1,2,3 or 4)
                                                                To see the results from other models don't provide this parameter.

    kkp.bin - data file that was analysed
    
    GaussianModel(folder):  contains results for each part of the main project scope

                            Part1:  dpluspeak_masshist.png
                                    fullscale_masshist.png

                            Part2:  dpluspeak_masshist_withfit.png
                                    fitparams.txt
                            
                            Part3:  fitparams.txt
                                    fullscale_masshist_withfit.png
                            
                            Part4:  residuals.png

    SumOfGaussiansModel(folder):            fitparams.txt                   - contains the best fit parameters for the given model
                                            fullscale_masshist_withfit.png  - contains a histogram of the mass distribution with the fit
                                            residuals.png                   - plot of the residuals
    
    CrystalBallModel(folder):               fitparams.txt   
                                            fullscale_masshist_withfit.png
                                            residuals.png
    
    SumOfCrystalBallGaussianModel(folder):  fitparams.txt
                                            fullscale_masshist_withfit.png
                                            residuals.png

    mu1_allmodels.csv - contains mean mass values from each of the models for the 1st peak with their statistical errors
    mu2_allmodels.csv - contains mean mass values from each of the models for the 2nd peak with their statistical errors

    Results_NoCrystalBall(folder):      when Crystal Ball was not included in computation of systematic errors due to model choice:
                                        NoCrystalBall_mu1_differences.csv - differences in mu_1 values from different models
                                        NoCrystalBall_mu2_differences.csv - differences in mu_2 values from different models
                                        NoCrystalBall_results_with_Estat_Esys.txt  - final results for the D masses with statistical ans systematic errors

    Results_WithCrystalBall(folder):    all models included in computation of systematic errors due to model choice
                                        WithCrystalBall_mu1_differences.csv
                                        WithCrystalBall_mu2_differences.csv
                                        WithCrystalBall_results_with_Estat_Esys.txt
    
    MomentumRapidityStudy(folder):      plot_D0mass_vs_momentum.png - mass vs momentum plot for D0
                                        plot_D0mass_vs_rapidity.png - mass vs rapidity plot for D0
                                        plot_sigma_vs_momentum.png - width of the D0 and D+ peak vs momentum plot
                                        plot_sigma_vs_rapidity.png - width of the D0 and D+ peak vs rapidity plot
                                        plot_D+mass_vs_momentum.png - mass vs momentum plot for D+
                                        plot_D+mass_vs_rapidity.png - mass vs rapidity plot for D+

    GaussianModel.py:                   Gaussian model Class with 4 parts (4 main scope assignment parts)
    SumOfGaussiansModel.py:             Sum of Gaussians Model Class - outputs files in its respective folder (mentioned above)
    CrystalBallModel.py:                Crystal Ball Model Class - outputs files in its respective folder
    SumOfCrystalBallGaussianModel.py:   Sum of Crystal Ball and Gaussian Class - outputs files in its respective folder

    calc_differences.py: calculates systematic errors - outputed:  Results_NoCrystalBall folder with files
                                                                    Results_WithCrystalBall folder with files:
                                                                 
    MomentumRapidityStudy.py: outputs the files in MomentumRapidityStudy folder
