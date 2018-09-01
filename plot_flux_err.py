import glob
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit


basepath="/work/00115/gebhardt/maverick/detect/20180717v011/lines/20180717v011*/"
basename="20180717v011_*specf.dat"


#basepath="/work/00115/gebhardt/maverick/detect/20180717v011/lines/20180717v011_1/"
#basename="20180717v011_*specf.dat"




def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None

    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
#here ... y would mean the minimum noise value (never 0)?


def fit_gaussian(central,x,y):
    #a bit different in that I am being aggressive and will force the A high enough to cover the max-value
    parm, pcov = curve_fit(gaussian, x, y)


fig = plt.figure(figsize=(16,9))
plt.title("Flux_Err by Wavelength for 20180717v011")
plt.xlabel("wavelength")
plt.ylabel("flux_err")


#todo: also collect the ifu and amp info
#todo: and assign colors based on that
iter = 0
allzero = []
badwave = []
bad = []
huge_unc = []
bad_ifu_info = [] #just ifuid and amp

unc_matrix = [] #each row is one uncertainty spectrum
wave_matrix = []

fit_parms = [] #wavelenth, x0, sigma, A, y

for f in glob.glob(basepath+basename):
    iter += 1
    try:
        out = np.loadtxt(f, dtype=None)

        wavelength = np.array(out[:, 0])

        if wavelength[0] == wavelength[1]:
            badwave.append(f)
            continue

        fluxerr = np.array(out[:, 2])  # * 1e-17

        if len(fluxerr[np.where(fluxerr != 0)]) == 0: #all zeroes
            print("**** ALL ZERO *****")
            allzero.append(f)
            continue

        if np.max(fluxerr) > 10:
            #todo: what ifu,amp?
            #in l2 file ...
            #date      ob       spc  slt ifu   fib
            #20180717 011 multi 051 105 051 LU 074 20180717T052650.4 3505.21
            d = op.dirname(f)
            out = np.loadtxt(op.join(d,"l2"),dtype=str,usecols=(3,4,5,6,7))
            huge_unc.append((np.max(fluxerr),f,out))

            for l in out:
                amp = l[3]
                ifuid = l[2]

                if not (ifuid,amp) in bad_ifu_info:
                    bad_ifu_info.append((ifuid,amp))

            continue #skip this one

            #huge_unc.append((np.max(fluxerr),f))
        #counterr = out[:, 4]  #if want to plot this later

        unc_matrix.append(fluxerr)
        wave_matrix.append(wavelength)
        plt.scatter(wavelength,fluxerr,s=0.5)
        print("(%d)" %(iter),f)
    except:
        print("Failed: ",f)
        bad.append(f)


print ("ALL ZEROS ....")
for a in allzero:
    print(a)

print("\n\nHUGE UNCERTAINTIES ....")
for a in huge_unc:
    print(a)

print("\n\nBAD IFU SUMMARY ....")
for a in bad_ifu_info:
    print(a)

plt.savefig("flux_err.png")
plt.show()


um = np.array(unc_matrix)
wm = np.array(wave_matrix)
rows,cols = wm.shape
fit_parms = []
for c in range(cols):
    print(c)
    plt.close('all')
    v,e,_ = plt.hist(um[:,c],bins='auto')
    e = e[1:] #bin edges, just drop the 0 so we skew slightly positive, but that is okay for my purpose

    #parm: 0 = x0,  1 = sigma, 2 = Area , 3 = y
    width = e[1]-e[0]
    x0 = e[np.argmax(v)]
    parm, pcov = curve_fit(gaussian, e, v, p0=(x0,0.2,20.,0.0),
                           bounds=((x0 - width, 0.01, 5.0, 0.0),
                                   (x0 + width, np.inf, np.inf, np.inf)))

    x = np.linspace(0,e[-1],100)
    plt.plot(x,gaussian(x,parm[0],parm[1],parm[2],parm[3]))
    plt.title("wave=%g\nx0=%f sigma=%f\nArea=%f y=%f"%(int(wm[0,c]), parm[0], parm[1], parm[2],parm[3]))
    fit_parms.append((wm[0,c],parm[0], parm[1], parm[2],parm[3]))
    plt.tight_layout()
    plt.savefig("unc_w%d.png" %(int(wm[0,c])))


with open("unc_fit_parms.txt","w") as f:
    f.write("wave x0 sigma Area y\n")
    for l in fit_parms:
        f.write("%f %f %f %f %f" %(l[0],l[1],l[2],l[3],l[4]))
        f.write("\n")

#print(fit_parms)

    #todo: fit each to gaussian and record parameters



