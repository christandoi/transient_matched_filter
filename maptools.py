import numpy as np
import re, os, time
from math import floor
from spt3g import core, maps, mapspectra, beams, sources, cluster
from spt3g.simulations import foregrounds as fg
from scipy.signal import convolve2d

def sqdeg(ra1,ra2,dec1,dec2):
    steradians = ((ra1-ra2)*np.pi/180)*(np.sin((np.pi/180)*dec1)-np.sin((np.pi/180)*dec2)) 
    square_degrees = steradians/(np.pi/180)**2
    return square_degrees

def ra_to_hour(ra):
    hh=ra//15
    curr_deg = ra%15
    mm = floor(curr_deg/15*60)
    curr_deg = curr_deg/15*60 - mm
    ss = round(curr_deg*60,4)
    if ss<10.:
        ss = f'0{ss}'
    return f'{int(hh)} {mm} {ss}'

def hour_to_deg(hh, mm, ss):
    tot_min = mm + ss/60
    tot_hour = hh + tot_min/60
    deg = tot_hour*15
    return deg

def dec_to_ddmmss(dec):
    dd=floor(np.abs(dec))
    curr_deg = np.abs(dec) - dd
    mm = floor(curr_deg*60)
    curr_deg = curr_deg*60 - mm
    ss = round(curr_deg*60,4)
    if ss<10.:
        ss = f'0{ss}'
    if dec<0.:
        dd*=-1
    return f'{int(dd)} {mm} {ss}'

def ddmmss_to_deg(dd, mm, ss):
    neg=False
    if dd<0.:
        neg=True
    dd = np.abs(dd)
    tot_min = mm + ss/60
    tot_deg = dd + tot_min/60
    if neg:
        tot_deg *= -1
    return tot_deg

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out=func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} completed in {round(end-start)}s')
        
        return out
    return wrapper

def timer_dec(dec, logging):
    def decorator(func):
        if not logging:
            return func
        return dec(func)
    return decorator

def get_obsid(file):
     match = re.search(r"[0-9]{8,9}", file)
     obsid = int(match.group(0))
     return obsid if match is not None else None 

def sample_field(field, n=1, **pad_kwargs):
    """Sample n points uniformly in the specified field"""
    (lo_ra, hi_ra), (lo_dec, hi_dec) = sources.get_field_extent(field, **pad_kwargs)
    if lo_ra > hi_ra:
        lo_ra -= 360 * core.G3Units.deg
    if lo_dec > hi_dec:
        core.log_fatal("dec bounds must be in increasing order")
    rand_ra = np.random.uniform(lo_ra, hi_ra, size=n)
    rand_dec = np.arcsin(np.random.uniform(np.sin(lo_dec), np.sin(hi_dec), size=n))
    if n == 1:
        return (rand_ra[0], rand_dec[0])
    return (rand_ra, rand_dec)

def rename_source_id(frame):
    if frame.type != core.G3FrameType.Map:
        return
    band = re.search(r"\d{2,3}GHz", frame["Id"]).group(0)
    del frame["Id"]
    ra = frame["T"].alpha_center
    dec = frame["T"].delta_center
    newid = sources.radec_to_spt_name(ra, dec, "S")
    frame["Id"] = "{}-{}".format(newid, band)
    if (
        "005548-6123.6" in frame["Id"]
        or "235845-6052.9" in frame["Id"]  # special winter
        or "051926-4545.9" in frame["Id"]  # special summer a
    ):  # the problem duplicates
        del frame["Id"]  # delete the id we just gave it
        from astropy.coordinates.angles import (Angle,)  # copy the code from radec_to_spt_name
        ra = np.mod(ra, 360 * core.G3Units.deg)
        opts = dict(sep="", pad=True, precision=3)
        rastr = Angle(ra / core.G3Units.rahour, "hour").to_string(**opts)
        decstr = Angle(dec / core.G3Units.deg, "deg").to_string(alwayssign=True, **opts)
        decr = "{:.3f}".format(dec / core.G3Units.arcmin).split(".")[1]
        decstr = decstr[:5] + ".{}".format(decr[:1])
        newid = "SPT-{} J{}{}".format("S", rastr, decstr)
        frame["Id"] = "{}-{}".format(newid, band)

def extract_thumbnail(frame, pixel, width = 1.2*core.G3Units.deg, source=None):
    '''
    extracts a small thumbnail at a given location in a map
    '''
    if frame.type != core.G3FrameType.Map:
        return
    if 'Wpol' in frame:
        core.log_error(
            "Only works on unpolarized maps. " "Making maps unpolarized.",
            unit="extract_thumbnail",
        )
        frame = maps.MakeMapsUnpolarized(frame)

    t = frame['T']
    width_px = round(width / t.res)
    x, y = round(pixel % t.shape[1]), round(pixel // t.shape[1])
   
    thumb_t = t.extract_patch(x, y, width_px, width_px)
    thumb_w = maps.G3SkyMapWeights()
    thumb_w.TT = frame['Wunpol'].TT.extract_patch(x, y, width_px, width_px)
    thumb_frame = core.G3Frame(core.G3FrameType.Map)
    thumb_frame['T'] = thumb_t
    thumb_frame['Wunpol'] = thumb_w

    for k in frame.keys():
        if k not in ['T', 'Wunpol']:
            thumb_frame[k] = frame[k]
    if source:
        thumb_frame['Source'] = source

    return thumb_frame

def getmapframe(f):
    '''
    quick method to search through g3 files and find the map frame
    '''
    for fr in cluster.GridFile(f):
        if 'Id' in fr:
            if 'Wpol' in fr:
                maps.MakeMapsUnpolarized(fr)
                return fr
            if 'Wunpol' in fr:
                return fr

def sub(fr1,fr2,fac=1):
    '''
    quick method to make a difference map
    '''
    maps.RemoveWeights(fr1, zero_nans=True)
    maps.RemoveWeights(fr2, zero_nans=True)
    assert(not(np.isnan(fr1['T']).any() or np.isnan(fr2['T']).any()))
    out = core.G3Frame(core.G3FrameType.Map)
    out['Id'] = fr1['Id']
    diff = fr1['T'] - float(fac)*fr2['T']
    diff *= (fr1['Wunpol'].TT > 0)
    diff.weighted = False
    out['T'] = diff
    out['Wunpol'] = fr1['Wunpol']
    return out

def nan_trim(T_map):
    '''
    removes NaN values from input maps, trimming to a rectangular cutout around the data
    '''
    arrT = np.array(T_map)
    rows_trimmed = arrT[~np.isnan(arrT).all(axis=1), :]
    all_trimmed = rows_trimmed[: , ~np.isnan(rows_trimmed).all(axis=0)]
    return all_trimmed

def get_lxly(mapparams):
    '''
    gets lx, ly for input map parameters: nx, ny give the number of pixels and dx, dy gives
    the map resolution
    '''
    nx, ny, dx, dx = mapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dx))
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def LminusR_diff(subfield, obsid, band, cutout_info):
    '''
    creates a difference map in a given observations using left going scans and right going scans
    '''
    subfield_dec = cutout_info[subfield]['dec']
    subfield_width = cutout_info[subfield]['width']
    dx = 0.25
    height = 2400
    frames_list = []
    
    file = f'/sptgrid/analysis/tau_deconv_ptsrc/{subfield}/{obsid}_{band}GHz_tonly.g3.gz'
    for frame in core.G3File(file):
        if 'Id' in frame and frame['Id'] == f'Left{band}GHz':
            left_frame = frame
        if 'Id' in frame and frame['Id'] == f'Right{band}GHz':
            right_frame = frame
        del frame

    maps.RemoveWeights(left_frame, zero_nans=True)
    maps.RemoveWeights(right_frame, zero_nans=True)
    lr_diff = mapspectra.map_analysis.subtract_two_maps(left_frame, right_frame, divide_by_two=True)

    proj0_T = maps.FlatSkyMap(subfield_width, height, dx*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=0,delta_center=subfield_dec*core.G3Units.deg)
    maps.reproj_map(lr_diff['T'], proj0_T)

    proj0_w = maps.FlatSkyMap(subfield_width, height, dx*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=0,delta_center=subfield_dec*core.G3Units.deg)
    maps.reproj_map(lr_diff['Wunpol'].TT, proj0_w)
    del lr_diff

    frames_list.append(proj0_T)
    del proj0_T
    frames_list.append(proj0_w)
    del proj0_w

    frames_dict = {}

    frames_dict['t'] = frames_list[0]
    frames_dict['w'] = frames_list[1]
    frames_dict['a'] = mapspectra.map_analysis.apodmask.make_border_apodization(frames_dict['w'])

    return frames_dict

def LplusR_diff(subfield, obsid, band, cutout_info):
    '''
    adds left and right scans to create a full map
    '''
    subfield_dec = cutout_info[subfield]['dec']
    subfield_width = cutout_info[subfield]['width']
    dx = 0.25
    height = 2400
    frames_list = []
    
    file = f'/sptgrid/analysis/tau_deconv_ptsrc/{subfield}/{obsid}_{band}GHz_tonly.g3.gz'
    for frame in core.G3File(file):
        if 'Id' in frame and frame['Id'] == f'Left{band}GHz':
            left_frame = frame
        if 'Id' in frame and frame['Id'] == f'Right{band}GHz':
            right_frame = frame
        del frame

    maps.RemoveWeights(left_frame, zero_nans=True)
    maps.RemoveWeights(right_frame, zero_nans=True)
    lr_diff = mapspectra.map_analysis.add_two_maps(left_frame, right_frame)

    proj0_T = maps.FlatSkyMap(subfield_width, height, dx*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=0,delta_center=subfield_dec*core.G3Units.deg)
    maps.reproj_map(lr_diff['T'], proj0_T)

    proj0_w = maps.FlatSkyMap(subfield_width, height, dx*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=0,delta_center=subfield_dec*core.G3Units.deg)
    maps.reproj_map(lr_diff['Wunpol'].TT, proj0_w)
    del lr_diff

    frames_list.append(proj0_T)
    del proj0_T
    frames_list.append(proj0_w)
    del proj0_w

    frames_dict = {}

    frames_dict['t'] = frames_list[0]
    frames_dict['w'] = frames_list[1]
    frames_dict['a'] = mapspectra.map_analysis.apodmask.make_border_apodization(frames_dict['w'])

    return frames_dict

def find_map_data(frame, width_pad=500, height_pad=500):
    '''
    Finds the first and last rows/columns of data, then 
    calculates height and width (with optional padding)
    for reprojection purposes.
    Returns dictionary with height,width (after padding),
    center x and y, and first/last rows/columns of data.
    '''
    map_data = {}
    find_xy = np.asarray(frame['T']).copy()
    y0 = np.argmax((~np.isnan(find_xy)).sum(axis=1) > 0) #find first row with data in it
    y1 = find_xy.shape[0] - np.argmax((~np.isnan(np.flipud(find_xy))).sum(axis=1) > 0) #work backwards to find the last row with data in it
    height = y1-y0
    center_y = (height/2)+y0
    height += height_pad
    x0 = np.argmax((~np.isnan(find_xy)).sum(axis=0) > 0) #find first row with data in it
    x1 = find_xy.shape[1] - np.argmax((~np.isnan(np.fliplr(find_xy))).sum(axis=0) > 0) #work backwards to find the last row with data in it
    width = x1-x0
    center_x = (width/2) +x0
    width += width_pad
    map_data['center_y'] = int(center_y)
    map_data['center_x'] = int(center_x)
    map_data['height'] = int(height)
    map_data['width'] = int(width)
    map_data['x0'] = int(x0)
    map_data['x1'] = int(x1)
    map_data['y0'] = int(y0)
    map_data['y1'] = int(y1)
    return map_data

def resize_and_create_difference_map(map_file, coadd_file, output=None, map_data=None):
    '''
    gets cutouts of the same location in two maps (single observation, coadded observations),
    and then creates a difference map of those cutouts
    '''
    proj5_map = getmapframe(map_file)
    if map_data is None:
        maps.RemoveWeights(proj5_map, zero_nans=False)
        map_data = find_map_data(proj5_map)
        maps.ApplyWeights(proj5_map)
    coadd_map = getmapframe(coadd_file)
    #saves 30s (~25%) by extracting first then differencing 
    for map_frame in [proj5_map, coadd_map]:
        tmap = map_frame.pop('T')
        map_frame['T'] = tmap.extract_patch(map_data['center_x'], map_data['center_y'], map_data['width'], map_data['height'])
        wmap = map_frame.pop('Wunpol')
        map_frame['Wunpol'] = maps.G3SkyMapWeights()
        map_frame['Wunpol'].TT = wmap.TT.extract_patch(map_data['center_x'], map_data['center_y'], map_data['width'], map_data['height'])
    proj5_diff_map = mapspectra.map_analysis.subtract_two_maps(proj5_map, coadd_map)
    diff_wmap = proj5_diff_map.pop('Wunpol')
    del diff_wmap
    proj5_diff_map['Wunpol'] = maps.G3SkyMapWeights()
    proj5_diff_map['Wunpol'].TT = proj5_map['Wunpol'].TT
    if output is not None:
        wr = core.G3Writer(output)
        wr(proj5_diff_map)
        del wr
    else:
        return proj5_diff_map, map_data

def find_cutout_ra(frame, center_dec, cutout_width=2400, edge_pad=100, overlap=240):
    '''
    searches for the Right Ascension (RA) values to make overlapping cutouts of a map for a given output size
    '''
    p0_map = maps.FlatSkyMap(19200, cutout_width, .25*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=0, delta_center=center_dec)
    maps.reproj_map(frame['T'], p0_map)
    np.asarray(p0_map)[np.asarray(p0_map)==0.] = None
    p0_frame = core.G3Frame(core.G3FrameType.Map)
    p0_frame['T'] = p0_map
    map_data_p0 = find_map_data(p0_frame, width_pad=0, height_pad=0)
    # print(map_data_p0)
    center = map_data_p0['center_x']
    first = map_data_p0['x0'] + int(cutout_width/2) - edge_pad
    last = map_data_p0['x1'] - int(cutout_width/2) + edge_pad
    # print(first, last)
    cutouts_pos = list(np.arange(center, last, int(cutout_width-overlap)))
    cutouts_neg = list(np.arange(center, first, -int(cutout_width-overlap))[::-1])
    cutout_x_list = np.unique(np.asarray([first] + cutouts_neg + cutouts_pos + [last]))
    # print(cutout_x_list)
    cutout_ra_list = np.asarray([p0_frame['T'].xy_to_angle(int(x), map_data_p0['y1'])[0] for x in cutout_x_list])
    return cutout_ra_list

def create_cutout(skymap, center_ra, center_dec, cutout_size=2400):
    '''
    simple function to create a cutout and then reproject from a Zenith Equal Area projection (proj5) to the
    Sanson-Flamsteed projection (proj0)
    '''
    proj5_map = skymap.copy()
    proj0_map = maps.FlatSkyMap(cutout_size, cutout_size, .25*core.G3Units.arcmin, proj=maps.MapProjection.Proj0, alpha_center=center_ra, delta_center=center_dec)
    maps.reproj_map(proj5_map, proj0_map)
    
    return proj0_map

def beam_deconvolve_map(map_frame, bandstr, source_mask=None, input_beams='/path/to/file/beam_file.npz'):
    '''
    deconvolves the input beam, in Fourier space, for a proj5 map
    '''
    maps.ApplyWeights(map_frame)
    maps.RemoveWeights(map_frame, zero_nans=True)
    border_mask = mapspectra.map_analysis.apodmask.make_border_apodization(map_frame['Wunpol'].TT)
    proj5_apod_mask = border_mask
    proj5_apod_mask *= source_mask
    proj5_ellgrid = mapspectra.basicmaputils.get_lxly(parent=map_frame['T'], real=False)
    proj5_ellgrid = np.hypot(*proj5_ellgrid)
    beam_mask = proj5_ellgrid>14130 #this is where the 90GHz beam reaches 2%
    bl_map_2d = beams.beam_analysis.beam2d_from_beam1d(bandstr=f'{bandstr}GHz', parent=map_frame['T'], filename=input_beams, real=False) 
    map_res = map_frame['T'].res
    fft_proj5_map = mapspectra.basicmaputils.map_to_ft(map_frame['T'], apod_mask=proj5_apod_mask, res=map_res, real=False)
    beam_deconvolved_fft_proj5_map = fft_proj5_map/bl_map_2d
    beam_deconvolved_fft_proj5_map[beam_mask] = 0.
    beam_deconvolved_proj5_map = mapspectra.basicmaputils.ft_to_map(beam_deconvolved_fft_proj5_map, apod_mask=proj5_apod_mask, res=map_res)
    return beam_deconvolved_proj5_map

def rolling_window_2d(array, window_height=5, window_width=5, mode='same', boundary='fill', var=True):
    '''
    calculates a rolling variance across a 2D array, with modifiable size
    '''
    kernel_size = window_height*window_width
    mean_kernel = np.ones((window_height,window_width))/kernel_size
    mean_kernel[int(window_height/2), int(window_width/2)] = 0
    mean_of_sq = convolve2d(array**2, mean_kernel, mode=mode, boundary=boundary, fillvalue=0)
    mean = convolve2d(array, mean_kernel, mode=mode, boundary=boundary, fillvalue=0)
    rolling_variance = mean_of_sq - mean**2
    if var:
        return rolling_variance
    else:
        return mean

def get_radio_freq_dep(
    freq, freq0=150.0 * core.G3Units.GHz, spec_index_rg=-0.9, null_highfreq_radio=True
):
    '''
    used for converting between flux (mJy) and temperature (uK) at radio frequencies
    '''
    nr = fg.get_planck_db_dt(freq0)
    dr = fg.get_planck_db_dt(freq)
    epsilon_nu1_nu0 = nr / dr
    scaling = (freq / freq0) ** spec_index_rg
    radio_sed = epsilon_nu1_nu0 * scaling
    if null_highfreq_radio and (freq / core.G3Units.GHz > 230.0):
        radio_sed = 0.0
    return radio_sed

def injected_temp(t0, freq0=95, freqs=[95,150,220], alpha=-0.76):
    '''
    returns the scaled temperatures (Tcmb units) for 95, 150, 220 for a given spectral index
    '''
    uK_scalings = [get_radio_freq_dep(freq*core.G3Units.GHz, freq0=freq0*core.G3Units.GHz, spec_index_rg=alpha) for freq in freqs]
    temperatures = [t0 * scaling for scaling in uK_scalings]
    return temperatures

def linlogrange(min_amp, max_amp, step):
    range_arr =  np.array([np.arange(1,10,step)/(10**exp) for exp in np.arange(np.abs(min_amp),np.abs(max_amp),-1)]).flatten()
    if range_arr[-1] > 10**max_amp:
        range_arr = range_arr[range_arr<10**max_amp]
    if range_arr[-1] < 10**max_amp:
        range_arr = np.append(range_arr, 10**max_amp)
    return range_arr

def load_coadd(season='winter', band=90, filtered=True):
    coadd_dir = '/sptlocal/analysis/transients/coadds'
    
    if season == 'winter':
        years = '2019-2023'
    elif 'summer' in season:
        years = '2019-2022'
    elif 'wide' in season:
        years = 'yearAB'
    elif season == 'edfs':
        years = '24nov24'
    else:
        raise ValueError("must contain a valid SPT-3G season")
    
    filt_status = 'tonly'
    if filtered:
        filt_status += '_transient_filtered'
    file = f"{coadd_dir}/map_coadd_{band}GHz_{season}_{years}_{filt_status}.g3.gz"

    coadd = getmapframe(file)

    return coadd

def save_multiband_fits(map_frames, filename, overwrite=False):
    '''
    Assumes map_frames is in form of [90, 150, 220] temperature maps. Converts from spt3g_software to the
    FITS file format.
    '''
    import astropy.io.fits
    from spt3g.maps.fitsio import create_wcs_header
    hdulist = astropy.io.fits.HDUList()

    header_data = map_frames[0]
    header = create_wcs_header(header_data)
    bitpix = {
        np.dtype(np.int16): 16,
        np.dtype(np.int32): 32,
        np.dtype(np.float32): -32,
        np.dtype(np.float64): -64,
    }
    header['BITPIX'] = bitpix[np.asarray(header_data).dtype]
    header['NAXIS'] = 2 
    header['MAPTYPE'] = 'FLAT'
    header['COORDREF'] = str(header_data.coord_ref)
    header['POLAR'] = False
    header['UNITS'] = str(header_data.units)
    header['WEIGHTED'] = False
    hdu = astropy.io.fits.ImageHDU(np.asarray([np.asarray(map_frame) for map_frame in map_frames]), header=header)
    hdu.header['ISWEIGHT'] = False
    hdu.header['POLTYPE'] = 'I'
    hdu.header['OVERFLOW'] = header_data.overflow
    hdu.header['FLATPOL'] = header_data.flat_pol
    hdu.header['CTYPE3'] = 'FREQ'
    hdulist.append(hdu)
    del hdu

    if overwrite:
        if os.path.exists(filename):
            os.remove(filename)
    else:
        assert(not os.path.exists(filename))

    hdulist.writeto(filename)
