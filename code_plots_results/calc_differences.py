# calc_differences.py
# code developed by Katarina Zatkova, Una Alberti
# part of the DAH F2 Project: Make Accurate Measurements of Particle Masses

# Calculation of differences between model values for systematic error determination
# Storing final values in a file

import csv
import numpy as np

models = np.array(["Gaussian Model", "Sum-Of-Gaussians Model", "Crystal Ball Model", "Sum-Of-CrystalBall-Gaussian Model"])
mu_1 = np.array([1.8694476305376722, 1.869444912179654, 1.8695043726772904, 1.8694438461420697])
mu_1_err = np.array([1.981226131090311e-05, 4.089066828769925e-05, 1.9427527006676826e-05, 4.871444068102057e-05])
mu_2 = np.array([1.9681007717223389, 1.9681158770943394, 1.9682338091301088, 1.968069507806066])
mu_2_err = np.array([1.0100016447411172e-05, 1.7372949112472225e-05, 9.870270467171594e-06, 2.0185534150714976e-05])

diff_mu1 = []
diff_mu2 = []

indices = np.array([0, 1, 3])

with open('mu1_differences.csv', 'w') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerow("1 - Gaussian Model")
    writer.writerow("2 - Sum-Of-Gaussians Model")
    writer.writerow("3 - Crystal Ball Model")
    writer.writerow("4 - Sum-Of-CrystalBall-Gaussian Model")
    for i in indices:
        for j in indices:
            writer.writerow([abs(mu_1[i]-mu_1[j]), i+1, j+1])
            diff_mu1.append([abs(mu_1[i]-mu_1[j])])
    f1.close()

with open('mu2_differences.csv', 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    writer.writerow("1 - Gaussian Model")
    writer.writerow("2 - Sum-Of-Gaussians Model")
    writer.writerow("3 - Crystal Ball Model")
    writer.writerow("4 - Sum-Of-CrystalBall-Gaussian Model")
    for i in indices:
        for j in indices:
            writer.writerow([abs(mu_2[i]-mu_2[j]), i+1, j+1])
            diff_mu2.append([abs(mu_2[i]-mu_2[j])])
    f1.close()

diff_mu1 = np.asarray(diff_mu1)
diff_mu2 = np.asarray(diff_mu2)

max_diff_mu1 = np.max(diff_mu1)
max_diff_mu2 = np.max(diff_mu2)

with open('results_with_Estat_Esys.txt', 'w') as f:
    print("\n", file=f)
    print("Model:                              Mean       =   Value     ±   Estat     ±   Esys", file=f)
    print("The First Peak", file=f)
    print(str(models[0]) + ":                     μ" + "          =   " + "{0:1.7f}".format(mu_1[0]) + "  ±   " + "{0:1.7f}".format(mu_1_err[0]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1), file=f)
    print(str(models[1]) + ":             μ" + "          =   " + "{0:1.7f}".format(mu_1[1]) + "  ±   " + "{0:1.7f}".format(mu_1_err[1]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1), file=f)
    print(str(models[2]) + ":                 μ" + "          =   " + "{0:1.7f}".format(mu_1[2]) + "  ±   " + "{0:1.7f}".format(mu_1_err[2]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1), file=f)
    print(str(models[3]) + ":  μ" + "          =   " + "{0:1.7f}".format(mu_1[3]) + "  ±   " + "{0:1.7f}".format(mu_1_err[3]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1), file=f)

print("\n")
print("Model:                              Mean       =   Value     ±   Estat     ±   Esys")
print("The First Peak")
print(str(models[0]) + ":                     μ" + "          =   " + "{0:1.7f}".format(mu_1[0]) + "  ±   " + "{0:1.7f}".format(mu_1_err[0]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1))
print(str(models[1]) + ":             μ" + "          =   " + "{0:1.7f}".format(mu_1[1]) + "  ±   " + "{0:1.7f}".format(mu_1_err[1]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1))
print(str(models[2]) + ":                 μ" + "          =   " + "{0:1.7f}".format(mu_1[2]) + "  ±   " + "{0:1.7f}".format(mu_1_err[2]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1))
print(str(models[3]) + ":  μ" + "          =   " + "{0:1.7f}".format(mu_1[3]) + "  ±   " + "{0:1.7f}".format(mu_1_err[3]) + "  ±   " + "{0:1.7f}".format(max_diff_mu1))

with open('results_with_Estat_Esys.txt', 'a') as f:
    print("\n", file=f)
    print("The Second Peak", file=f)
    print(str(models[0]) + ":                     μ" + "          =   " + "{0:1.6f}".format(mu_2[0]) + "  ±   " + "{0:1.6f}".format(mu_2_err[0]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2), file=f)
    print(str(models[1]) + ":             μ" + "          =   " + "{0:1.6f}".format(mu_2[1]) + "  ±   " + "{0:1.6f}".format(mu_2_err[1]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2), file=f)
    print(str(models[2]) + ":                 μ" + "          =   " + "{0:1.6f}".format(mu_2[2]) + "  ±   " + "{0:1.6f}".format(mu_2_err[2]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2), file=f)
    print(str(models[3]) + ":  μ" + "          =   " + "{0:1.6f}".format(mu_2[3]) + "  ±   " + "{0:1.6f}".format(mu_2_err[3]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2), file=f)

print("\n")
print("The Second Peak")
print(str(models[0]) + ":                     μ" + "          =   " + "{0:1.6f}".format(mu_2[0]) + "  ±   " + "{0:1.6f}".format(mu_2_err[0]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2))
print(str(models[1]) + ":             μ" + "          =   " + "{0:1.6f}".format(mu_2[1]) + "  ±   " + "{0:1.6f}".format(mu_2_err[1]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2))
print(str(models[2]) + ":                 μ" + "          =   " + "{0:1.6f}".format(mu_2[2]) + "  ±   " + "{0:1.6f}".format(mu_2_err[2]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2))
print(str(models[3]) + ":  μ" + "          =   " + "{0:1.6f}".format(mu_2[3]) + "  ±   " + "{0:1.6f}".format(mu_2_err[3]) + "  ±   " + "{0:1.6f}".format(max_diff_mu2))
print("\n")
