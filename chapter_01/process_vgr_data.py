#
#  file:  process_vgr_data.py
#
#  Turn the Voyager 1 plasma and lecp data into random bytes
#
#  RTK, 25-Mar-2022
#  Last update:  25-Mar-2022
#
################################################################

import os
import numpy as np

def MakeBits0(A, threshold):
    z = A - threshold*np.median(A)
    w = []
    for i in range(len(z)):
        w.append(1 if z[i] > 0 else 0)
    b = []
    i = 0
    while (i < len(w)-1):
        if (w[i] != w[i+1]):
            b.append(w[i])
        i += 2
    return b

def MakeBytes0(A):
    b0 = MakeBits0(A, 0.9)
    b1 = MakeBits0(A, 1.0)[::-1]
    b2 = MakeBits0(A, 1.1)
    b = np.array(b0+b1+b2, dtype="uint8")
    n = len(b)//8
    c = np.array(b[:8*n]).reshape((n,8))
    z = []
    for i in range(n):
        t = (c[i] * np.array([128,64,32,16,8,4,2,1])).sum()
        z.append(t)
    return np.array(z).astype("uint8")

sb = MakeBytes0(np.load("plasma/v1_proton_speed_1977_1980.npy"))
db = MakeBytes0(np.load("plasma/v1_proton_density_1977_1980.npy"))
tb = MakeBytes0(np.load("plasma/v1_proton_thermal_1977_1980.npy"))

v = np.load("plasma/v2_keys_2007_2018.npy")[:,13]  # Voyager 2 plasma density 2007-2018
vb = MakeBytes0(v)

w = np.load("plasma/v2_keys_2007_2018.npy")[:,15]  # Voyager 2 w column
wb = MakeBytes0(w)[::-1]

b = np.hstack((sb,db,tb,vb,wb))
b.tofile("voyager_plasma_data.bin")

def MakeBits1(A):
    A[np.where(A < 0)] = np.nan
    A[np.where(np.isnan(A))] = np.nanmean(A)
    b0 = np.zeros(len(A), dtype="uint8")
    b0[np.where(A > 0.8*np.median(A))] = 1
    b1 = np.zeros(len(A), dtype="uint8")
    b1[np.where(A > 1.0*np.median(A))] = 1
    b2 = np.zeros(len(A), dtype="uint8")
    b2[np.where(A > 1.2*np.median(A))] = 1
    b = np.hstack((b0,b1,b2))
    w = []
    k = 0
    while (k < len(b)-1):
        if (b[k] != b[k+1]):
            w.append(b[k])
        k += 2
    return w

def ProcessFile(fname):
    v = np.loadtxt(fname, skiprows=2)
    return MakeBits1(v[:,4])

b = []
for year in range(1978,2022):
    b += ProcessFile("lecp/cosmic/v1_%4d_eb05_rate_1d.txt" % year)

n = len(b)//8
c = np.array(b[:8*n]).reshape((n,8))
z = []
for i in range(n):
    t = (c[i] * np.array([128,64,32,16,8,4,2,1])).sum()
    z.append(t)
z = np.array(z).astype("uint8")
z.tofile("voyager_cosmic_flux.bin")

def MakeBits2(A):
    A[np.where(A < 0)] = np.nan
    A[np.where(np.isnan(A))] = np.nanmean(A)
    b0 = np.zeros(len(A), dtype="uint8")
    b0[np.where(A > 0.8*np.median(A))] = 1
    b1 = np.zeros(len(A), dtype="uint8")
    b1[np.where(A > 1.0*np.median(A))] = 1
    b2 = np.zeros(len(A), dtype="uint8")
    b2[np.where(A > 1.2*np.median(A))] = 1
    b = np.hstack((b0,b1,b2))
    w = []
    k = 0
    while (k < len(b)-1):
        if (b[k] != b[k+1]):
            w.append(b[k])
        k += 2
    return w

def ProcessFile2(fname):
    v = np.loadtxt(fname, skiprows=2)
    b0 = MakeBits2(v[:,5])
    b1 = MakeBits2(v[:,7])
    b2 = MakeBits2(v[:,9])
    b2 = MakeBits2(v[:,11])
    b3 = MakeBits2(v[:,13])
    b4 = MakeBits2(v[:,15])
    b5 = MakeBits2(v[:,17])
    return b0 + b1[::-1] + b2 + b3[::-1] + b4 + b5[::-1]

b = []
for year in range(1977,2022):
    b += ProcessFile2("lecp/ion/v1_%4d_ion_flux_1h.txt" % year)

n = len(b)//8
c = np.array(b[:8*n]).reshape((n,8))
z = []
for i in range(n):
    t = (c[i] * np.array([128,64,32,16,8,4,2,1])).sum()
    z.append(t)
z = np.array(z).astype("uint8")
z.tofile("voyager_ion_flux.bin")

def MakeBits3(A):
    A[np.where(A < 0)] = np.nan
    A[np.where(np.isnan(A))] = np.nanmean(A)
    b0 = np.zeros(len(A), dtype="uint8")
    b0[np.where(A > 0.8*np.median(A))] = 1
    b1 = np.zeros(len(A), dtype="uint8")
    b1[np.where(A > 1.0*np.median(A))] = 1
    b2 = np.zeros(len(A), dtype="uint8")
    b2[np.where(A > 1.2*np.median(A))] = 1
    b = np.hstack((b0,b1,b2))
    w = []
    k = 0
    while (k < len(b)-1):
        if (b[k] != b[k+1]):
            w.append(b[k])
        k += 2
    return w

def ProcessFile3(fname):
    v = np.loadtxt(fname, skiprows=2)
    b0 = MakeBits3(v[:,5])
    b1 = MakeBits3(v[:,7])
    b2 = MakeBits3(v[:,9])
    return b0 + b1[::-1] + b2

b = []
for year in range(1977,2022):
    b += ProcessFile3("lecp/proton/v1_%4d_prot_flux_1h.txt" % year)

n = len(b)//8
c = np.array(b[:8*n]).reshape((n,8))
z = []
for i in range(n):
    t = (c[i] * np.array([128,64,32,16,8,4,2,1])).sum()
    z.append(t)
z = np.array(z).astype("uint8")
z.tofile("voyager_proton_flux.bin")

os.system("cat voyager_cosmic_flux.bin voyager_ion_flux.bin voyager_proton_flux.bin voyager_plasma_data.bin >voyager_plasma_lecp.bin")

