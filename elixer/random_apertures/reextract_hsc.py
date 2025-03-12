"""
this is meant to be executed from the working directory with the hsc_g coord files

You need to update which file is used

You need to specify local or ffsky or rescor in the code

"""

import sys
import os.path as op
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
from astropy.stats import sigma_clip
import astropy.io.misc.hdf5 as hdf5
import astropy.units as u
import copy
import glob
import pickle
import gc
import traceback

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G
from elixer import weighted_biweight as weighted_biweight
from elixer import mcmc_gauss
import astropy.stats.biweight as biweight

from tqdm import tqdm, trange
from itertools import combinations, product
from scipy.ndimage import gaussian_filter1d

import requests

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#form is integer dateobs (no 'v')
if "--input" in args:
    i = args.index("--input")
    try:
        inputfile = sys.argv[i + 1]
    except:
        print("no input file")
        exit(-1)
else:
    print("no input file")
    exit(-1)
#
# if "--output" in args:
#     i = args.index("--output")
#     try:
#         outputfile = sys.argv[i + 1]
#     except:
#         print("no output file")
#         exit(-1)
# else:
#     print("no output file")
#     exit(-1)

if "--sky":
    i = args.index("--sky")
    try:
        skychoice = int(sys.argv[i + 1])
    except:
        print("no sky")
        exit(-1)
else:
    print("no sky")
    exit(-1)

dT_name = f"{inputfile}_dets" #_ffrc"#.fits"
fT_name = f"{inputfile}_fibers" #_ffrc" #.fits"

if skychoice == 0: #local
    ffsky = False
    rescor = False
    dT_name += "_ll"
    fT_name += "_ll"
elif skychoice == 1: #local
    ffsky = True
    rescor = False
    dT_name += "_ff"
    fT_name += "_ff"
elif skychoice == 2: #local
    ffsky = True
    rescor = True
    dT_name += "_rc"
    fT_name += "_rc"
else:
    print("invalid sky")
    exit(-1)

#selection = sel #sel_bright
survey_name = "hdr5"

#hub
#rescor_path = "/home/jovyan/Hobby-Eberly-Telesco/hdr5/reduction/ffsky_rescor/"#rc{str(shot)[0:8]}v{str(shot)[8:]}.h5"):
#not hub
rescor_path = "/scratch/projects/hetdex/hdr5/reduction/ffsky_rescor/"#rc{str(shot)[0:8]}v{str(shot)[8:]}.h5"):

radius = 3.5

fiber_flux_offset = None
apply_mask = True
get_dust = True

multiprocess = True #but not using this switch here ... is diectly True/False in call

save_every = 1100 #so never, give the 1000 limit on the call from the outside
bin_every = None #3 * save_every - 1 #the -1 so it trips properly

#old#
#dets,shots = np.loadtxt("reextract.coords",dtype=int,unpack=True, usecols=[0,3])
#ras,decs,seeing,thruput = np.loadtxt("reextract.coords",dtype=float,unpack=True, usecols=[1,2,4,5])

#new
#ra           dec         shot     (wave) det   hsc id           flux(ujy) error  mag   brigt faint
#34.49354912 -5.16855732 20201014026 0 98000001 37484842471937319 988.15  216.97 23.91 23.70 24.18
#34.49354912 -5.16855732 20201020016 0 98000002 37484842471937319 988.15  216.97 23.91 23.70 24.18


shots,dets = np.loadtxt(inputfile,dtype=int,unpack=True, usecols=[2,4])
ras,decs   = np.loadtxt(inputfile,dtype=float,unpack=True, usecols=[0,1])

survey = Survey('hdr5')
survey_table = survey.return_astropy_table(return_good=False)

# define a pair of tables ... one for the base detection info and one for all the fiber data

#todo: what all might we want in the future?
def make_dT():
    return Table(dtype=[('detectid',np.int64), ('ra', float), ('dec', float), ('z', float), 
                  ('shotid', int), ('seeing',float),('response',float),('apcor',float),
                  ('sky_type',int),
                        
                  ('sn',float),('plya_classification',float),('combined_plae',float),
                  ('z_elixer',float),('best_pz',float),('elixer_flags',np.int32),('flag_best',np.int64),   
                  ('lum_lya',float),('lum_lya_err',float), 
                        
                  ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                  ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('dust_corr', (float, len(G.CALFIB_WAVEGRID))), #per fiber or per detection?
                ])
def make_fT():
    return Table(dtype=[('detectid',np.int64), ('fiber_id',str),
                  ('ra', float), ('dec', float),('fiber_ra', float), ('fiber_dec', float),
                  ('shotid', int),
                  ('raw_weight',float),
                  ('norm_weight',float),
                  ('sky_type',int), #0=local, 1=ffsky, 2=ffsky w/rescor
                  ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                  ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('wave_weights', (float, len(G.CALFIB_WAVEGRID))),
                  ('wave_masks', (float, len(G.CALFIB_WAVEGRID))),
                  ('clean_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                  ('clean_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ])

#def rescor_exists()



#
# iterate over the 
#



dT = make_dT()
fT = make_fT()

combined_dT = make_dT()
combined_fT = make_fT()

rescor_missed = []
bad_rescor_shots = []
missed_shots = []


if ffsky:
    if rescor:
        sky_type = 2
    else:
        sky_type = 1
else:
    sky_type = 0
    
apt = None

if bin_every is not None:
    bin_ctstr = "_001"
else:
    bin_ctstr = ""

bt = 0

#RESTARTING!!!
start_idx = 0
#start_idx = np.where(dets==9100524143001)[0][0] + 1
#bin_ctstr = "_007"

print(f"(re)starting at {start_idx} with {dets[start_idx]} ...")

#for ct, row in enumerate(tqdm(source_table[selection][0:10])):
#for ct, data in enumerate(tqdm(zip(dets[start_idx:],ras[start_idx:],decs[start_idx:],shots[start_idx:],seeing[start_idx:],thruput[start_idx:]),total=len(dets[start_idx:]))):
#for ct, data in enumerate(tqdm(zip(dets,ras,decs,shots,seeing,thruput),total=len(dets))):
#testing single, with tqdm
#for ct, data in enumerate(tqdm(zip(dets, ras, decs, shots), total=len(dets))):
for ct, data in enumerate(zip(dets, ras, decs, shots)): #w/o tqdm
    try:
        #det,ra,dec,shot,seeing_fwhm,tput = data[0],data[1],data[2],data[3],data[4],data[5]
        det, ra, dec, shot  = data[0], data[1], data[2], data[3]

        sel = survey_table['shotid'] == shot
        if np.count_nonzero(sel) != 1:
            print(f"Could not find shot {shot}")
            missed_shots.append(shot)
            continue

        tput = survey_table['response_4540'][sel][0]
        seeing_fwhm = survey_table['fwhm_virus'][sel][0]
        
        if rescor:
            if not op.isfile(f"{rescor_path}rc{str(shot)[0:8]}v{str(shot)[8:]}.h5"):
                print(f"{det} rescor does not exist for {shot}")
                rescor_missed.append(det)
                bad_rescor_shots.append(shot)
                continue
            
        bt += 1
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        #print(coord,row['detectid'],row['shotid'])
        dust_corr = deredden_spectra(G.CALFIB_WAVEGRID,coord)
        

        
        apt = get_spectra(coord, survey=survey_name, shotid=shot,
                              ffsky=ffsky, multiprocess=True,
                              rad=radius, tpmin=0.0, fiberweights=False, #fiber wieght is redundant with fiber_info
                              return_fiber_info=True,
                              loglevel="ERROR", fiber_flux_offset = fiber_flux_offset,
                              ffsky_rescor=rescor, apply_mask=apply_mask)
        
        
        if len(apt) != 1:
            print(f"Problem. Wrong number of detection spectra: {det}, {len(apt)}")
            continue
            
        if apt['fiber_info'].shape == (1,0):
            print(f"Problem with fiber_info ... not returned.")
            continue
            
        #get residual for reference .. will get this later and apply as wanted
#         empty_fiber, empty_fiber_err, contrib, status = SU.get_empty_fiber_residual(hdr=survey_name,
#                                                                                     rtype=rtype,
#                                                                                     shotid=row['shotid'],
#                                                                                     seeing=None,
#                                                                                     response=None,
#                                                                                     ffsky=ffsky,
#                                                                                     add_rescor=rescor,
#                                                                                     persist=True)
        
        sum_weight = np.sum(apt['fiber_info'][0][:,4].astype(float))
        
        shot_fiber_table = get_fibers_table(shot=shot,coords=coord,survey=survey_name,radius=radius,astropy=True)
        #,multiframe=)
        
        for fiber,weights,masks,clean_fluxd,clean_fluxd_err in zip(apt['fiber_info'][0],
                                                                   apt['fiber_wavelength_weights'][0],
                                                                   apt['fiber_wavelength_masks'][0],
                                                                   apt['clean_fluxd'][0],
                                                                   apt['clean_fluxd_err'][0]):
            #get the match in the shot_fiber_table
            fiber_id = fiber[0]
            
            sel_fiber = shot_fiber_table['fiber_id'] == fiber_id
            if np.count_nonzero(sel_fiber) != 1:
                print(f"Problem: did not match {fiber_id}")
                continue
            
            multiframe = fiber[1]
            ra = float(fiber[2])
            dec = float(fiber[3])
            raw_weight = float(fiber[4])
            norm_weight = raw_weight / sum_weight
            
            #get the fiber spectra
            if sky_type == 0:
                fluxd = shot_fiber_table['calfib'][sel_fiber]
            else:
                fluxd = shot_fiber_table['calfib_ffsky'][sel_fiber]
            fluxe = shot_fiber_table['calfibe'][sel_fiber]
            
            #per wavelength weights
            wave_weights = weights
            wave_masks = masks
                        
            
            #add to fibers table         
            fT.add_row([det,fiber_id, ra, dec,ra,dec,shot,
                        raw_weight,norm_weight, sky_type, fluxd, fluxe,wave_weights,wave_masks,clean_fluxd,clean_fluxd_err])

        dT.add_row([det,ra, dec, -1, shot,
                    seeing_fwhm,tput,np.nanmedian(apt['apcor'][0]),
                    sky_type,
                    
                    -1,-1,-1,
                    -1,-1,-1,-1,
                    -1,-1,
                                            
                    
                    apt['spec'][0] ,apt['spec_err'][0], dust_corr])
        
        try:
            del apt
            del dust_corr
        except:
            print(f"Exception deleting apt")
        
    except Exception as e:
        print(f"{det} Exception", e)
        print(traceback.format_exc())
        
                
    if ct % save_every == 0:

        try:
            if len(combined_fT) == 0:
                combined_fT = fT
                combined_dT = dT
            else:
                combined_fT = vstack([combined_fT,fT])
                combined_dT = vstack([combined_dT,dT])

            combined_dT.write(dT_name+bin_ctstr+".fits",format='fits',overwrite=True)
            combined_fT.write(fT_name+bin_ctstr+".fits",format='fits',overwrite=True)
        except:
            print(f"Exception vstack...")
            print(traceback.format_exc())

        try:
            del dT
            del ft
            gc.collect()
        except:
            pass

        #make new tables
        dT = make_dT()
        fT = make_fT()
        
        
        if bin_every is not None and bt >= bin_every:
            #we just wrote out combined_dt, so it is safe to iterate
            del combined_fT
            del combined_dT
            bin_ctstr = "_"+str(int(bin_ctstr[1:])+1).zfill(3)
            combined_dT = make_dT() #make new table
            combined_fT = make_fT() #make new table
            bt = 0
            print(f"starting new bin_ctstr: {bin_ctstr}; last included ct = {ct} ({ct+start_idx}), last included det = {det}")
        #else:
        #    print(f"No write: bin_every = {bin_every}, bt = {bt}")

#final write
if len(combined_fT) == 0:
    combined_fT = fT
    combined_dT = dT
else:
    combined_fT = vstack([combined_fT,fT])
    combined_dT = vstack([combined_dT,dT])

combined_fT.sort('detectid')      
combined_dT.sort('detectid')  

combined_fT.add_index('detectid')
combined_dT.add_index('detectid')

combined_dT.write(dT_name+bin_ctstr+".fits",format='fits',overwrite=True)
combined_fT.write(fT_name+bin_ctstr+".fits",format='fits',overwrite=True)
        
