import numpy as np
import pandas as pd
from spt3g import core, maps, sources, mapspectra
from spt3g.mapspectra import basicmaputils as utils
from spt3g.mapspectra import map_analysis
import maptools
import tmf
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm
import time

"""
    =========
    LOAD MAPS
    =========
    
    Given an input catalog of stars, find the specific star of interest (as given by unique star_observation_id)
    and load in the relevant maps to use in the matched filter.
"""
stars = pd.read_csv('/path/to/file.csv')
output_dir = '/path/to/file/'
star_observation_id = 176194338
matching = stars[stars.obsid==star_observation_id]
star_subfield = matching.Field.values[0]
subfield_center_dec = float(star_subfield.split('dec')[1])*core.G3Units.deg
pt_src_file = sources.get_field_source_list(star_subfield, analysis='online')
new_star_90_file = f'/path/{star_subfield}/{star_observation_id}_90GHz_tonly.g3.gz'
new_star_150_file = f'/path/{star_subfield}/{star_observation_id}_150GHz_tonly.g3.gz'
new_star_220_file = f'/path/{star_subfield}/{star_observation_id}_220GHz_tonly.g3.gz'
coadd_90_file = '/path/to/file/map_coadd_90GHz_winter_2019-2023_tonly.g3.gz'
coadd_150_file = '/path/to/file/map_coadd_150GHz_winter_2019-2023_tonly.g3.gz'
coadd_220_file = '/path/to/file/map_coadd_220GHz_winter_2019-2023_tonly.g3.gz'
start = time.time()

"""
    =====================
    INPUT MAP PREPARATION
    =====================

    Takes the input maps (single observation, coadded observations in all 3 observing bands) and does
    the necessary steps in creating maps ready for the matched filter:
    - adjust size and create difference map
    - masks bright static point sources (to avoid filtering artifacts in Fourier space)
    - deconvolve the beam from the ZEA/proj5 map (necessary when reprojecting to Sanson-Flamsteed/proj0)
    - creates cutouts of beam deconvolved maps as well as the non-beam-deconvolved maps, which are used for
      estimating the white noise
"""
# create beam convolved maps
print(f'making difference maps, {round(time.time()-start)}s')
# run 150 first to get map_data (size/locations/pixels/etc)
proj5_150_differenced, proj5_map_data = maptools.resize_and_create_difference_map(new_star_150_file, coadd_150_file)
proj5_90_differenced, _ =  maptools.resize_and_create_difference_map(new_star_90_file, coadd_90_file, map_data=proj5_map_data)
proj5_220_differenced, _ = maptools.resize_and_create_difference_map(new_star_220_file, coadd_220_file, map_data=proj5_map_data)
# deconvolve beam
print(f'deconvolving beam, {round(time.time()-start)}s')
proj5_source_mask = map_analysis.apodmask.make_apodized_ptsrc_mask(proj5_150_differenced['T'], point_source_file=pt_src_file)
proj5_90_beam_dc_differenced = maptools.beam_deconvolve_map(proj5_90_differenced, '90GHz', source_mask=proj5_source_mask)
proj5_150_beam_dc_differenced = maptools.beam_deconvolve_map(proj5_150_differenced, '150GHz', source_mask=proj5_source_mask)
proj5_220_beam_dc_differenced = maptools.beam_deconvolve_map(proj5_220_differenced, '220GHz', source_mask=proj5_source_mask)
# run a quick reprojection to find the suitable RAs for cutouts
print(f'finding cutouts, {round(time.time()-start)}s')
maps.ApplyWeights(proj5_90_differenced)
maps.RemoveWeights(proj5_90_differenced, zero_nans=False)
cutouts_ra = maptools.find_cutout_ra(proj5_90_differenced, subfield_center_dec)
# we use the beam convolved maps for white noise estimation
cutouts_90_for_wn_t = []
cutouts_150_for_wn_t = []
cutouts_220_for_wn_t = []
cutouts_90_beam_dc_t = []
cutouts_150_beam_dc_t = []
cutouts_220_beam_dc_t = []
cutouts_90_beam_dc_w = []
cutouts_150_beam_dc_w = []
cutouts_220_beam_dc_w = []

print(f'making cutouts, {round(time.time()-start)}s')
for ra in tqdm(cutouts_ra):
    cutouts_90_for_wn_t.append(maptools.create_cutout(proj5_90_differenced['T'], ra, subfield_center_dec))
    cutouts_150_for_wn_t.append(maptools.create_cutout(proj5_150_differenced['T'], ra, subfield_center_dec))
    cutouts_220_for_wn_t.append(maptools.create_cutout(proj5_220_differenced['T'], ra, subfield_center_dec))
    cutouts_90_beam_dc_t.append(maptools.create_cutout(proj5_90_beam_dc_differenced, ra, subfield_center_dec))
    cutouts_150_beam_dc_t.append(maptools.create_cutout(proj5_150_beam_dc_differenced, ra, subfield_center_dec))
    cutouts_220_beam_dc_t.append(maptools.create_cutout(proj5_220_beam_dc_differenced, ra, subfield_center_dec))
    cutouts_90_beam_dc_w.append(maptools.create_cutout(proj5_90_differenced['Wunpol'].TT, ra, subfield_center_dec))
    cutouts_150_beam_dc_w.append(maptools.create_cutout(proj5_150_differenced['Wunpol'].TT, ra, subfield_center_dec))
    cutouts_220_beam_dc_w.append(maptools.create_cutout(proj5_220_differenced['Wunpol'].TT, ra, subfield_center_dec))

del proj5_90_differenced
del proj5_150_differenced
del proj5_220_differenced
del proj5_90_beam_dc_differenced
del proj5_150_beam_dc_differenced
del proj5_220_beam_dc_differenced

"""
    ========================
    INTIALIZATION PARAMETERS
    ========================

    Mostly map/beam info and the necessary conversions to Fourier space. Very much a WIP so these values change
    frequently during testing.
"""
nbands=3
input_beams = '/path/to/file.npz'
beam_lmax = np.where(np.load(input_beams)['90']<.02)[0][0] #new lmax, cut where 90 GHz beam power is 2%
bl_1d_dict = {f'{bl_1d}GHz': np.load(input_beams)[bl_1d] for bl_1d in np.load(input_beams) if bl_1d != "ell"} #cut the ells
bl_1d_dict['one'] = np.ones(shape=beam_lmax)
proj0_y, proj0_x = cutouts_150_beam_dc_t[0].shape
cutout_ellgrid = utils.get_lxly(parent=cutouts_150_beam_dc_t[0], real=False)
extent_val = [np.min(cutout_ellgrid[0]), np.max(cutout_ellgrid[0]), np.min(cutout_ellgrid[1]), np.max(cutout_ellgrid[1])]
lx_bins, ly_bins = cutout_ellgrid
lx_bin = cutout_ellgrid[0][0,1] - cutout_ellgrid[0][0,0]
ly_bin = cutout_ellgrid[1][1,1] - cutout_ellgrid[1][0,0]
cutout_ellgrid = np.hypot(*cutout_ellgrid)
ellmask = cutout_ellgrid<=(beam_lmax+1.) #here is where we'll impose the 2% beam cut after the map has been made
ells = np.arange(1,beam_lmax+1)
wn_ellmask = np.logical_and(cutout_ellgrid>=(4000), cutout_ellgrid<=(10000))
tf_lmax = 20000 #note this is different than the map lmax of 14130
proj0_tf_map_2d = sources.fitting.construct_tf((proj0_y,proj0_x), .25*core.G3Units.arcmin, 500, tf_lmax, cm="wafer")
tf_threshold = 0.367 #.367 corresponds to kx~550
hpf_ell_idx = np.argwhere(proj0_tf_map_2d[0,:]>tf_threshold)[0][0]
# hpf_ell = hpf_ell_idx * lx_bin ## todo: why is this not working? a hpf at ell~500 should have no problems
hpf_ell = 750 #we don't lose much info pushing up to higher ell, not ideal but after some rough testing it shouldn't matter *too* much
proj0_tf_hpf_ellmask = np.abs(lx_bins)<=(hpf_ell)
#source profile
profile_2d = mapspectra.basicmaputils.interp_cl_2d(np.ones(shape=(20000)), .25*core.G3Units.arcmin, (proj0_y, proj0_x), real=False)
signal_template = profile_2d * proj0_tf_map_2d
signal_template[proj0_tf_hpf_ellmask] = 0.
signal_template[~ellmask] = 0.
signal_template[np.isnan(signal_template)] = 0.
signal_template[signal_template == 0.] = None;
#for modeling noise later
min_amp = -12
max_amp = -2
amp_bins = 30
amp_arr = np.logspace(min_amp, max_amp, amp_bins)
#for temp scaling later
temp_scaling_low = maptools.injected_temp(t0=1, freq0=95, freqs=[150,220], alpha=-3)
temp_scaling_high = maptools.injected_temp(t0=1, freq0=95, freqs=[150,220], alpha=3)
alpha_90150_range=np.linspace(temp_scaling_low[0], temp_scaling_high[0], 5)
alpha_90220_range=np.linspace(temp_scaling_low[1], temp_scaling_high[1], 5)
 
"""
    ============================
    FUNCTIONS FOR NOISE MODELING
    ============================

    These consist of functions that take in various maps and estimate their noise in Fourier space, and also their noise correlations in the form of auto-spectra (i.e. 90X90, 150X150, 220x220) and cross-spectra (90X150, 90X220, 150X220).
"""
def noise_PSD_knee_beam_dc(wn_estimate, knee_estimate, alpha_estimate=4):
    '''
    Models a noise Power Spectral Density from a given white noise estimate and ell_knee estimate. Assumes noise
    purely as the combination of white (flat power at all angular scales) and atmospheric (1/f pink noise) where the ell_knee is defined as the angular scale where the atmospheric noise power as twice that of the white noise power;
    typically this is somewhere around ell=1500-2500 (approximately 5 arcminutes).

    This function takes in the beam convolved map and uses an initial guess at white noise to come up with a more accurate
    estimate.
    '''
    wn_1d_est = np.ones(shape=ells.shape)*wn_estimate
    atm_1d_est = np.nan_to_num((ells/knee_estimate)**-alpha_estimate,posinf=1e3)
    atm_1d_est *= wn_1d_est
    atm_1d_est[:500] = atm_1d_est[500] #doesn't matter since this gets cut to ell=500 at a minimum
    nl_1d_est = wn_1d_est + atm_1d_est
    nl_2d_est = mapspectra.basicmaputils.interp_cl_2d(nl_1d_est, .25*core.G3Units.arcmin, (proj0_y,proj0_x), real=False)
    nl_2d_est *= proj0_tf_map_2d**2
    nl_2d_est[proj0_tf_hpf_ellmask] = 0.
    nl_2d_est[~ellmask] = 0.
    nl_2d_est[np.isnan(nl_2d_est)] = 0.
    nl_2d_est[nl_2d_est == 0.] = None
    return nl_2d_est

def noise_PSD_amplitude(wn_estimate, atmo_amp_est=5, alpha_estimate=4, sample_ell=1512, bandstr='150GHz', bandstr2=None, beam_lmax=14130):
    '''
    Models a noise Power Spectral Density from a given white noise estimate and ell_knee estimate. Assumes noise
    purely as the combination of white (flat power at all angular scales) and atmospheric (1/f pink noise) where the ell_knee is defined as the angular scale where the atmospheric noise power as twice that of the white noise power;
    typically this is somewhere around ell=1500-2500 (approximately 5 arcminutes).

    This function takes in the beam deconvolved map and uses the accurate white noise estimate to model the true noise PSD.
    '''
    if bandstr2 is None:
        bandstr2 = bandstr
    wn_1d_est = np.ones(shape=ells.shape)*wn_estimate/bl_1d_dict[bandstr][:beam_lmax]/bl_1d_dict[bandstr2][:beam_lmax]
    atm_1d_est = np.nan_to_num((ells/sample_ell)**-alpha_estimate,posinf=1e3)
    atm_1d_est *= atmo_amp_est
    atm_1d_est[:500] = atm_1d_est[500] 
    nl_1d_est = wn_1d_est + atm_1d_est
    nl_2d_est = mapspectra.basicmaputils.interp_cl_2d(nl_1d_est, .25*core.G3Units.arcmin, (proj0_y,proj0_x), real=False)
    nl_2d_est *= proj0_tf_map_2d**2
    nl_2d_est[proj0_tf_hpf_ellmask] = 0.
    nl_2d_est[~ellmask] = 0.
    nl_2d_est[np.isnan(nl_2d_est)] = 0.
    nl_2d_est[nl_2d_est == 0.] = None
    return nl_2d_est

def model_auto(PSD_before_beam_dc, PSD_after_beam_dc, initial_wn_estimate, bandstr, min_amp=-11, max_amp=-5, amp_bins=30):
    '''
    Models the auto-spectra of input PSDs using a best-fit chi squared test.
    '''
    wn_fit_knee = 1500 #shouldn't matter since we mask ell<4000
    initial_noise_PSD_fit_wn_search = noise_PSD_knee_beam_dc(initial_wn_estimate, wn_fit_knee, 4)
    initial_noise_PSD_fit_wn_search[~wn_ellmask] = None
    psd_data_wn_search = PSD_before_beam_dc.copy()
    psd_data_wn_search[proj0_tf_hpf_ellmask] = None
    psd_data_wn_search[psd_data_wn_search == 0.] = None
    psd_data_wn_search[~wn_ellmask] = None
    ratio_mean = np.nanmean((psd_data_wn_search/initial_noise_PSD_fit_wn_search).flatten())
    best_fit_wn = initial_wn_estimate * ratio_mean
    atmo_amp_arr = np.logspace(min_amp, max_amp, amp_bins)
    chisq_list = []
    for amp_est in tqdm(atmo_amp_arr, leave=False):
        initial_noise_PSD_fit = noise_PSD_amplitude(best_fit_wn, amp_est, 4, bandstr=bandstr)
        psd_data_amp_search = PSD_after_beam_dc.copy()
        psd_data_amp_search[proj0_tf_hpf_ellmask] = None
        psd_data_amp_search[psd_data_amp_search == 0.] = None
        psd_data_amp_search[~ellmask] = None
        initial_noise_PSD_fit[initial_noise_PSD_fit == 0.] = None
        initial_noise_PSD_fit[~ellmask] = None
        chisq = np.nansum((psd_data_amp_search - initial_noise_PSD_fit)**2)
        chisq_list.append(chisq)
    chisq_arr = np.asarray(chisq_list)
    best_fit_amp_idx = np.argmin(chisq_arr)
    best_fit_amp = atmo_amp_arr[best_fit_amp_idx]
    return best_fit_wn, best_fit_amp, chisq_arr
    
def model_cross(cross_PSD, amp1, amp2, amp_bins=30):
    '''
    Models the cross-spectra of input PSDs using a best-fit chi squared test.
    '''
    cross_amp_arr = np.logspace(np.log10(np.min([amp1,amp2])/10), np.log10(np.max([amp1,amp2])*10), amp_bins)
    chisq_list = []
    for amp_est in tqdm(cross_amp_arr, leave=False):
        initial_noise_PSD_fit = noise_PSD_amplitude(0, amp_est, 4)
        # initial_noise_PSD_fit_amp_search = initial_noise_PSD_fit.copy()
        cross_psd_data_amp_search = cross_PSD.copy()
        cross_psd_data_amp_search[proj0_tf_hpf_ellmask] = None
        cross_psd_data_amp_search[cross_psd_data_amp_search == 0.] = None
        cross_psd_data_amp_search[~ellmask] = None
        initial_noise_PSD_fit[initial_noise_PSD_fit == 0.] = None
        initial_noise_PSD_fit[~ellmask] = None
        chisq = np.nansum((cross_psd_data_amp_search - initial_noise_PSD_fit)**2)
        chisq_list.append(chisq)
    chisq_arr = np.asarray(chisq_list)
    best_fit_amp_idx = np.argmin(chisq_arr)
    best_fit_amp = cross_amp_arr[best_fit_amp_idx]
    return best_fit_amp, chisq_arr

"""
    ==============
    MATCHED FILTER
    ==============

    Now that all of the input maps are correctly formatted, and components of the noise-correlation matrix have been
    modeled, we can run the matched filter to search for signals.
"""
n_cutouts = len(cutouts_90_for_wn_t)
cutout_snmaps_dict = {}
sn_dict = {}

for i in tqdm(range(n_cutouts)):
    ## grab cutouts
    print(f'grabbing cutouts, {round(time.time()-start)}s')
    cutout_90_for_wn_t = cutouts_90_for_wn_t[i]
    np.asarray(cutout_90_for_wn_t)[np.isnan(np.asarray(cutout_90_for_wn_t))] = 0.
    cutout_150_for_wn_t = cutouts_150_for_wn_t[i]
    np.asarray(cutout_150_for_wn_t)[np.isnan(np.asarray(cutout_150_for_wn_t))] = 0.
    cutout_220_for_wn_t = cutouts_220_for_wn_t[i]
    np.asarray(cutout_220_for_wn_t)[np.isnan(np.asarray(cutout_220_for_wn_t))] = 0.
    cutout_beam_dc_90_t = cutouts_90_beam_dc_t[i]
    np.asarray(cutout_beam_dc_90_t)[np.isnan(np.asarray(cutout_beam_dc_90_t))] = 0.
    cutout_beam_dc_150_t = cutouts_150_beam_dc_t[i]
    np.asarray(cutout_beam_dc_150_t)[np.isnan(np.asarray(cutout_beam_dc_150_t))] = 0.
    cutout_beam_dc_220_t = cutouts_220_beam_dc_t[i]
    np.asarray(cutout_beam_dc_220_t)[np.isnan(np.asarray(cutout_beam_dc_220_t))] = 0.
    cutout_90_beam_dc_w = cutouts_90_beam_dc_w[i]
    cutout_150_beam_dc_w = cutouts_150_beam_dc_w[i]
    cutouts220_beam_dc_w = cutouts_220_beam_dc_w[i]

    ## make masks. this is a necessary step to smooth edges so the data is well behaved in Fourier space
    print(f'making masks, {round(time.time()-start)}s')
    cutout_source_mask = map_analysis.apodmask.make_apodized_ptsrc_mask(cutout_beam_dc_150_t, point_source_file=pt_src_file)
    proj0_border_apod_mask_90 = map_analysis.apodmask.make_border_apodization(cutout_90_beam_dc_w)
    proj0_apod_mask_90 = proj0_border_apod_mask_90 * cutout_source_mask
    proj0_border_apod_mask_150 = map_analysis.apodmask.make_border_apodization(cutout_150_beam_dc_w)
    proj0_apod_mask_150 = proj0_border_apod_mask_150 * cutout_source_mask
    proj0_border_apod_mask_220 = map_analysis.apodmask.make_border_apodization(cutouts220_beam_dc_w)
    proj0_apod_mask_220 = proj0_border_apod_mask_220 * cutout_source_mask
    #getting values of .999999999 which breaks the SN map, fix it here
    proj0_apod_mask_90[proj0_apod_mask_90>=.99] = 1.
    proj0_apod_mask_150[proj0_apod_mask_150>=.99] = 1.
    proj0_apod_mask_220[proj0_apod_mask_220>=.99] = 1.
    full_apod = proj0_apod_mask_90*proj0_apod_mask_150*proj0_apod_mask_220

    ## make auto spectra (beam convolved and beam deconvolved maps)
    print(f'making autos, {round(time.time()-start)}s')
    cutout_90_PSD = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_90_for_wn_t, apod_mask=proj0_apod_mask_90, return_2d=True, real=False)['TT']
    cutout_150_PSD = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_150_for_wn_t, apod_mask=proj0_apod_mask_150, return_2d=True, real=False)['TT']
    cutout_220_PSD = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_220_for_wn_t, apod_mask=proj0_apod_mask_220, return_2d=True, real=False)['TT']
    cutout_90_PSD_beam_deconvolved = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_90_t, apod_mask=proj0_apod_mask_90, return_2d=True, real=False)['TT']
    cutout_150_PSD_beam_deconvolved = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_150_t, apod_mask=proj0_apod_mask_150, return_2d=True, real=False)['TT']
    cutout_220_PSD_beam_deconvolved = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_220_t, apod_mask=proj0_apod_mask_220, return_2d=True, real=False)['TT']

    ## make cross spectra (beam deconvolved maps only)
    print(f'making cross, {round(time.time()-start)}s')
    cutout_90150_PSD_raw = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_90_t, input2=cutout_beam_dc_150_t, apod_mask=proj0_apod_mask_90*proj0_apod_mask_150, return_2d=True, real=False)['TT']
    cutout_90150_PSD = cutout_90150_PSD_raw.copy()
    cutout_90150_PSD[~ellmask] = None
    cutout_90150_PSD[proj0_tf_hpf_ellmask] = None
    cutout_90220_PSD_raw = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_90_t, input2=cutout_beam_dc_220_t, apod_mask=proj0_apod_mask_90*proj0_apod_mask_220, return_2d=True, real=False)['TT']
    cutout_90220_PSD = cutout_90220_PSD_raw.copy()
    cutout_90220_PSD[~ellmask] = None
    cutout_90220_PSD[proj0_tf_hpf_ellmask] = None
    cutout_150220_PSD_raw = mapspectra.map_analysis.calculate_powerspectra(input1=cutout_beam_dc_150_t, input2=cutout_beam_dc_220_t, apod_mask=proj0_apod_mask_150*proj0_apod_mask_220, return_2d=True, real=False)['TT']
    cutout_150220_PSD = cutout_150220_PSD_raw.copy()
    cutout_150220_PSD[~ellmask] = None
    cutout_150220_PSD[proj0_tf_hpf_ellmask] = None
    
    ## find wn, model auto spectra
    print(f'modeling autos, {round(time.time()-start)}s')
    initial_wn_estimate_90 = np.mean(cutout_90_PSD[0,111:334]) #hardcoded 4000-12000 ell range for ell_bin=36.
    initial_wn_estimate_150 = np.mean(cutout_150_PSD[0,111:334])
    initial_wn_estimate_220 = np.mean(cutout_220_PSD[0,111:334])
    model_wn_90, model_amp_90, model_chisq_90_arr = model_auto(cutout_90_PSD, cutout_90_PSD_beam_deconvolved, initial_wn_estimate_90, min_amp=min_amp, max_amp=max_amp, amp_bins=amp_bins, bandstr='90GHz')
    model_wn_150, model_amp_150, model_chisq_150_arr = model_auto(cutout_150_PSD, cutout_150_PSD_beam_deconvolved, initial_wn_estimate_150, min_amp=min_amp, max_amp=max_amp, amp_bins=amp_bins, bandstr='150GHz')
    model_wn_220, model_amp_220, model_chisq_220_arr = model_auto(cutout_220_PSD, cutout_220_PSD_beam_deconvolved, initial_wn_estimate_220, min_amp=min_amp, max_amp=max_amp, amp_bins=amp_bins, bandstr='220GHz')
    noise_PSD_model_90 = noise_PSD_amplitude(model_wn_90, model_amp_90, bandstr='90GHz')
    noise_PSD_model_150 = noise_PSD_amplitude(model_wn_150, model_amp_150, bandstr='150GHz')
    noise_PSD_model_220 = noise_PSD_amplitude(model_wn_220, model_amp_220, bandstr='220GHz')

    ## model cross spectra
    print(f'modeling cross, {round(time.time()-start)}s')
    model_amp_90150, model_chisq_90150_arr = model_cross(cutout_90150_PSD, model_amp_90, model_amp_150)
    model_amp_90220, model_chisq_90220_arr = model_cross(cutout_90220_PSD, model_amp_90, model_amp_220)
    model_amp_150220, model_chisq_150220_arr = model_cross(cutout_150220_PSD, model_amp_150, model_amp_220)
    cross_90150_model = noise_PSD_amplitude(0, model_amp_90150)
    cross_90220_model = noise_PSD_amplitude(0, model_amp_90220)
    cross_150220_model = noise_PSD_amplitude(0, model_amp_150220)
    
    ## prep MF: create the covariance matrix and format input maps for the matched filter
    print(f'making MF inputs, {round(time.time()-start)}s')
    cov_mat = np.zeros(shape=(proj0_y,proj0_x,nbands,nbands))
    cov_mat[:,:,0,0] = noise_PSD_model_90
    cov_mat[:,:,1,1] = noise_PSD_model_150
    cov_mat[:,:,2,2] = noise_PSD_model_220
    cov_mat[:,:,0,1] = cov_mat[:,:,1,0] = cross_90150_model
    cov_mat[:,:,0,2] = cov_mat[:,:,2,0] = cross_90220_model
    cov_mat[:,:,1,2] = cov_mat[:,:,2,1] = cross_90150_model
    cov_mat[cov_mat==0.] = None
    inv_cov = np.linalg.inv(cov_mat)
    skymaps_arr = np.asarray([cutout_beam_dc_90_t, cutout_beam_dc_150_t, cutout_beam_dc_220_t])
    apod_arr = np.asarray([proj0_apod_mask_90, proj0_apod_mask_150, proj0_apod_mask_220])
    ft_maps = np.array([mapspectra.basicmaputils.map_to_ft(skymap, apod_mask=proj0_apod_mask, res=.25*core.G3Units.arcmin, real=False) for skymap, proj0_apod_mask in zip(skymaps_arr, apod_arr)])
    ft_maps_mat = np.stack(ft_maps, axis=2)
    
    ## run MF
    print(f'running MF, {round(time.time()-start)}s')
    cutout_snmaps_dict[i] = []
    for a90150 in tqdm(alpha_90150_range, desc='a90150 initial', leave=True):
        for a90220 in tqdm(alpha_90220_range, desc='a90220 initial', leave=False):
            temp_scaling = [1, a90150, a90220]
            multiband_signals = np.moveaxis(np.asarray([signal_template * temp_scale for temp_scale in temp_scaling]),0,2)[:,:,np.newaxis,:]
            multiband_signals_transpose = np.transpose(multiband_signals, axes=(0, 1, 3, 2))
            predicted_variance = 1/(np.nansum(multiband_signals * inv_cov * multiband_signals_transpose) / (proj0_y * proj0_x))
            matched_filter = (inv_cov * multiband_signals).sum(axis=(3)) * predicted_variance
            filtered_fft_map = np.nan_to_num(np.sum(matched_filter * ft_maps_mat, axis=2))
            filtered_fft_map = mapspectra.MapSpectrum2D(spec=filtered_fft_map, parent=cutout_beam_dc_150_t, real=False)
            filtered_map = mapspectra.basicmaputils.ft_to_map(filtered_fft_map, apod_mask=full_apod, res=.25*core.G3Units.arcmin)
            snmap = tmf.make_snmaps([filtered_map], full_apod, None, .25*core.G3Units.arcmin, None)[0]
            cutout_snmaps_dict[i].append(snmap)
    print(f'MF done on cutout {i+1}/{n_cutouts}, {round(time.time()-start)}s')

    ## once the MF is complete, we can search the signal-to-noise maps (SN maps) for signals of high significance, i.e.
    ## a transient source
    # gather all SN
    sn_cutoff = 7
    sn_dict[i] = []
    save_all_snmaps = False
    for snmap in cutout_snmaps_dict[i]:
        groups = sources.finding.find_groups(snmap, signoise=1, nsigma=sn_cutoff)
        if type(groups['maxvals']) == float:
            if groups['maxvals'] == 0.:
                sn_dict[i].append(0.)
            else:
                print('non-zero float ?')
                sn_dict[i].append(-1)
        elif len(groups['maxvals']) == 1:
            sn_dict[i].append(groups['maxvals'][0])
        elif len(groups['maxvals']) > 1:
            save_all_snmaps = True
        else:
            save_all_snmaps = True

    # check if there are any maps containing pixels above SN threshold
    sn_dict[i] = np.asarray(sn_dict[i])
    if sn_dict[i][sn_dict[i]>0.].any():
        print(f'cutout {i} detection above threshold: {round(np.max(sn_dict[i]),2)}')
        #find the source location in the highest SN temperature scaled map
        maxsn_snmap = sn_dict[i][np.argmax(sn_dict[i])]
        maxsn_groups = sources.finding.find_groups(maxsn_snmap, signoise=1, nsigma=sn_cutoff)
        #make a mask around that source to renormalize SN correctly
        trans_x = int(maxsn_groups['xcen'])
        trans_y = int(maxsn_groups['ycen'])
        transient_mask = np.ones(shape=(2400,2400))
        transient_mask[trans_y-9:trans_y+10,trans_x-9:trans_x+10] = 0.
        snmap_mask = full_apod*transient_mask

        ### search a grid of spectral indices and create a spline to find the spectral index of our transient source
        z = np.asarray(sn_dict[i]).copy()
        z = z.reshape(len(alpha_90150_range),len(alpha_90220_range))
        interp_spline = RectBivariateSpline(alpha_90150_range, alpha_90220_range, z, kx=4, ky=4)
        xfine = np.linspace(alpha_90150_range.min(), alpha_90150_range.max(), 20)
        yfine = np.linspace(alpha_90220_range.min(), alpha_90220_range.max(), 20)
        zinterp = interp_spline(xfine, yfine)
        xpeak_interp, ypeak_interp = np.unravel_index(zinterp.argmax(), zinterp.shape)
        interp_90150 = xfine[xpeak_interp]
        interp_90220 = yfine[ypeak_interp]

        temp_scaling = [1, interp_90150, interp_90220]
        multiband_signals = np.moveaxis(np.asarray([signal_template * temp_scale for temp_scale in temp_scaling]),0,2)[:,:,np.newaxis,:]
        multiband_signals_transpose = np.transpose(multiband_signals, axes=(0, 1, 3, 2))
        predicted_variance = 1/(np.nansum(multiband_signals * inv_cov * multiband_signals_transpose) / (proj0_y * proj0_x))
        peak_matched_filter = (inv_cov * multiband_signals).sum(axis=(3)) * predicted_variance
        filtered_fft_map = np.nan_to_num(np.sum(peak_matched_filter * ft_maps_mat, axis=2))
        filtered_fft_map = mapspectra.MapSpectrum2D(spec=filtered_fft_map, parent=cutout_beam_dc_150_t, real=False)
        filtered_map = mapspectra.basicmaputils.ft_to_map(filtered_fft_map, apod_mask=full_apod, res=.25*core.G3Units.arcmin)
        #here we impose our transient source mask on the interpolated temp scaling, only for SN calculations
        peak_snmap = tmf.make_snmaps([filtered_map], snmap_mask, None, .25*core.G3Units.arcmin, None)[0]
        peak_groups = sources.finding.find_groups(peak_snmap, signoise=1, nsigma=sn_cutoff)
        # print(peak_groups['maxvals'])
    
        wr = core.G3Writer(f'{output_dir}/snmaps/{star_observation_id}_cutout{i}_snmaps_dict.g3.gz')
        temp_fr = core.G3Frame(core.G3FrameType.Map)
        try:
            temp_fr['SN'] = peak_groups['maxvals'][0]
            temp_fr['SN_interpolated'] = True
        except TypeError as e:
            print(f'{e}. saving non-interpolated value')
            temp_fr['SN'] = np.max(sn_dict[i])
            temp_fr['SN_interpolated'] = False
        temp_fr['temp_90150'] = interp_90150
        temp_fr['temp_90220'] = interp_90220
        temp_fr['snmap'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), peak_snmap.copy(order='C'))
        temp_fr['mf_90'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), peak_matched_filter[:,:,0].copy(order='C'))
        temp_fr['mf_150'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), peak_matched_filter[:,:,1].copy(order='C'))
        temp_fr['mf_220'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), peak_matched_filter[:,:,2].copy(order='C'))
        temp_fr['cutout_90'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), skymaps_arr[0].copy(order='C'))
        temp_fr['cutout_150'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), skymaps_arr[1].copy(order='C'))
        temp_fr['cutout_220'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), skymaps_arr[2].copy(order='C'))
        wr(temp_fr)
        del wr

    else:
        #return a blank frame so we know there were no sources here
        wr = core.G3Writer(f'{output_dir}/snmaps/{star_observation_id}_cutout{i}_snmaps_dict.g3.gz')
        temp_fr = core.G3Frame(core.G3FrameType.Map)
        temp_fr['SN'] = 0.
        del wr

    # if we caught a bug/unknown result, save all the maps and look manually
    if save_all_snmaps:
        wr = core.G3Writer(f'{output_dir}/snmaps/{star_observation_id}_cutout{i}_snmaps_dict.g3.gz')
        for j, cutout_snmap in tqdm(enumerate(cutout_snmaps_dict[i]), leave=False):
            temp_fr = core.G3Frame(core.G3FrameType.Map)
            temp_fr['id'] = j
            temp_fr['snmap'] = maps.FlatSkyMap.array_clone(cutouts_150_beam_dc_t[i].copy(), cutout_snmap.copy(order='C'))
            wr(temp_fr)
        del wr