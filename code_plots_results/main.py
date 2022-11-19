# Imports
import sys
import numpy as np
from GaussianModel import GaussianModel
from SumOfGaussiansModel import SumOfGaussiansModel
from CrystalBallModel import CrystalBallModel
from SumOfCrystalBallGaussianModel import SumOfCrystalBallGaussianModel

# Main entry point of the program
if __name__ == "__main__":
    
    # reading input arguments
    args = sys.argv

    f = open("kkp.bin","r")
    b = np.fromfile(f,dtype=np.float32)
    ncol = 7
    # number of events
    nevent = len(b)/ncol
    xdata = np.split(b, nevent)
    # make list of invariant mass of events
    xmass = []
    for i in range(0, len(xdata)):
        xmass.append(xdata[i][0]/1000)

    if (len(args) == 3):
        
        model= str(args[1])
        part_n= int(args[2])  
      
        if model == "Gaussian":
            gaussianmodel = GaussianModel(xmass)
            
            if part_n == 1:
                gaussianmodel.part_1(True)

            elif part_n == 2:
                gaussianmodel.part_2()
            
            elif part_n == 3:
                gaussianmodel.part_3(True)

            elif part_n == 4:
                gaussianmodel.part_4()

            elif part_n == 5:
                txt = """This part deals with the enhancement of the project. We implemented 3 additional model functions:\n
                SumOfGaussians: PDF comprising a function which is the sum of two Gaussian functions.\n
                CrystalBall: Crystal Ball PDF which incorporates a non-Gaussian tail at the lower end of the mass peak.\n
                SumOfCrystalBallGaussian: PDF comprising a function which is the sum of a Gaussian function and a Crystal Ball function.\n
                \n
                To display the histogram with the fit and the residuals plot, together with the best fit parameters, for either of the models 
                provide their name as an input.
                \n 
                """
                print(txt)

            else:
                print("There are only 5 parts to this project.")
        
        else:
            print("To display results for the other models, don't provide the part number parameter.")


    elif (len(args) == 2 ):

        # When these models are chosen as input, the histogram with the fit and the residuals plots are outputed
        # the best fit parameters are also printed
        
        model= str(args[1])
        if model == "Gaussian":
            gaussianmodel = GaussianModel(xmass)
            gaussianmodel.part_3(True)
            gaussianmodel.part_4()

        elif model == "SumOfGaussians":
            sum_of_gaussiansmodel = SumOfGaussiansModel(xmass)
            sum_of_gaussiansmodel.fit(True)
            sum_of_gaussiansmodel.get_residuals()

        elif model == "CrystalBall":
            crystalballmodel = CrystalBallModel(xmass)
            crystalballmodel.fit(True)
            crystalballmodel.get_residuals()

        elif model == "SumOfCrystalBallGaussian":
            sum_of_crystalballgaussian = SumOfCrystalBallGaussianModel(xmass)
            sum_of_crystalballgaussian.fit(True)
            sum_of_crystalballgaussian.get_residuals()
        
        else:
            print("There are only 4 implemented models. Please choose between 'Gaussian', 'SumOfGaussians', 'CrystalBall' and 'SumOfCrystalBallGaussian'.\n")
            print("For the Gaussian model you can also choose a specific part of the project, e.g. 'python3 main.py Gaussian 2' would show Gaussian Model, 2nd part.")

    else:
        print("\n")
        print ("Usage: python3 main.py Model Part_number \n")
        print("There are 4 implemented models. Please choose between 'Gaussian', 'SumOfGaussians', 'CrystalBall' and 'SumOfCrystalBallGaussian'.\n")
        print("For the Gaussian model you can also choose a specific part of the project, e.g. 'python3 main.py Gaussian 2' would shows Gaussian Model, 2nd part. \n")

        sys.exit()
