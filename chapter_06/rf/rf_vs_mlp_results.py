import numpy as np
from scipy.stats import ttest_ind

mlp0 = np.load("rf_vs_mlp_mlp0.npy")
mlp1 = np.load("rf_vs_mlp_mlp1.npy")
rf0 = np.load("rf_vs_mlp_rf0.npy")
rf1 = np.load("rf_vs_mlp_rf1.npy")

print()
print("MLP, unscaled vs scaled (n=16):")
print("    unscaled mean+/-SE = %0.6f +/- %0.6f" % (mlp0.mean(), mlp0.std(ddof=1)/np.sqrt(len(mlp0))))
print("      scaled mean+/-SE = %0.6f +/- %0.6f" % (mlp1.mean(), mlp1.std(ddof=1)/np.sqrt(len(mlp1))))
_,p = ttest_ind(mlp0,mlp1)
print("    t-test p-value     = %0.12f" % p)
print()

print()
print("RF, unscaled vs scaled (n=16):")
print("    unscaled mean+/-SE = %0.6f +/- %0.6f" % (rf0.mean(), rf0.std(ddof=1)/np.sqrt(len(rf0))))
print("      scaled mean+/-SE = %0.6f +/- %0.6f" % (rf1.mean(), rf1.std(ddof=1)/np.sqrt(len(rf1))))
_,p = ttest_ind(rf0,rf1)
print("    t-test p-value     = %0.12f" % p)
print()

