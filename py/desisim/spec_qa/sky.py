"""
desisim.spec_qa.sky
============

Utility functions to do simple QA on the Sky Modeling
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys, os, pdb

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits

from desispec.resolution import Resolution
from desispec.log import get_logger
from desispec import util
from desispec.io import read_fiberflat
from desispec import interpolation as desi_interp

from desispec.io import frame as desi_io_frame
from desispec.io import fibermap as desi_io_fmap
from desispec.io import read_sky
from desispec import sky as dspec_sky
from desispec.fiberflat import apply_fiberflat

#from xastropy.xutils import xdebug as xdb

def tst_meansky_fibers(simspec_fil, frame_root, fflat_root, path=None):
    '''Examines mean sky in SKY fibers
    '''
    # imports

    # Truth from simspec
    simsp_hdu = fits.open(simspec_fil)
    hdu_names = [hdu.name for hdu in simsp_hdu]

    # DESI extraction
    # Load
    if path is None:
        path=''
    fiber_fil='fibermap'+frame_root[frame_root.find('-'):]
    fibermap = desi_io_fmap.read_fibermap(path+fiber_fil)

    # Loop on Camera
    #for camera in ['R','Z']:
    for camera in ['B','R','Z']:
        # Get DESIspec extractions
        frame_fil = 'frame-{:s}'.format(camera.lower())+frame_root
        frame = desi_io_frame.read_frame(path+frame_fil)

        # TRUTH (simspec)
        # Grab header
        idx = hdu_names.index('SKYPHOT_'+camera)
        hdr = simsp_hdu[idx].header
        # Generate wavelength array
        true_wave = hdr['CRVAL1'] + (hdr['CDELT1'])*np.arange(hdr['NAXIS1'])
        # Grab data
        true_sky = simsp_hdu[idx].data
        # Resample to extraction
        dw = frame.wave[1]-frame.wave[0]
        ww = frame.wave[0] + np.arange(len(frame.wave))*dw
        true_flux = desi_interp.resample_flux(ww, true_wave, true_sky)

        # Calibrate DESIspec extractions
        # Flat field
        fflat_fil = 'fiberflat-{:s}'.format(camera.lower())+fflat_root

        # read fiberflat
        fiberflat = read_fiberflat(path+fflat_fil)

        # apply fiberflat to sky fibers
        apply_fiberflat(frame, fiberflat)

        # Read sky
        sky_fil = 'sky-{:s}'.format(camera.lower())+frame_root
        skymodel=read_sky(path+sky_fil)

        # Isolate SKY fibers
        skyfibers=np.where((fibermap["OBJTYPE"]=="SKY")&
            (fibermap["FIBER"]>=frame.specmin)&(fibermap["FIBER"]<=frame.specmax))[0]

        # Apply Resolution to model
        cskytruth = np.zeros((len(skyfibers),len(frame.wave)))
        for i,skyfiber in enumerate(skyfibers):
            cskytruth[i] = frame.R[skyfiber].dot(true_flux)

        # Average up Truth
        avg_truth = np.mean(cskytruth,0) 

        # Average up Extracted Sky
        avg_desi_sky = np.mean(frame.flux[skyfibers,:],0)

        # subtract sky
        dspec_sky.subtract_sky(frame, skymodel)

        # Average up Extracted Sky
        avg_desi_sub = np.mean(frame.flux[skyfibers,:],0)

        # Generate QA
        if False:
            qa_mean_spec(frame.wave, avg_desi_sky, avg_desi_sub, None, 
                None, avg_truth, outfil='QA_sky_mean_'+camera+'.pdf')
            qa_fiber_chi(frame, skyfibers, outfil='QA_fiber_chi_'+camera+'.pdf')
        qa_fiber_stats(frame, skyfibers, outfil='QA_fiber_stats_'+camera+'.pdf')
        # 
        #break
        #xdb.set_trace()

def tst_deconvolve_mean_sky(simspec_fil, frame_root, fiber_fil, path=None):
    '''Compares deconvolved sky against Truth
    -- A bit goofy.  
    DEPRECATED
    '''
    # imports
    import specter.throughput as spec_thru
    import specter.psf.spotgrid as SpotGridPSF
    sys.path.append(os.path.abspath("/Users/xavier/DESI/desisim_v0.4.1/desisim/"))
    import interpolation as desi_interp
    import io as desisim_io
    #

    from astropy.io import fits
    from xastropy.xutils import xdebug as xdb

    # Truth from simspec
    simsp_hdu = fits.open(simspec_fil)
    hdu_names = [hdu.name for hdu in simsp_hdu]

    # DESI extraction
    # Load
    if path is None:
        path=''
    fiber_map = desi_io_fmap.read_fibermap(path+fiber_fil)

    # Loop on Camera
    #for camera in ['R','Z']:
    for camera in ['B','R','Z']:
        # TRUTH FIRST
        # Grab header
        idx = hdu_names.index('SKYPHOT_'+camera)
        hdr = simsp_hdu[idx].header
        # Generate wavelength array
        wave = hdr['CRVAL1'] + (hdr['CDELT1'])*np.arange(hdr['NAXIS1'])
        # Grab data
        sky_truth = simsp_hdu[idx].data

        # Get model
        frame_fil = 'frame-{:s}'.format(camera.lower())+frame_root
        obs_frame = desi_io_frame.read_frame(path+frame_fil)
        skymodel, skyflux, skyvar = dspec_sky.compute_sky(obs_frame,fiber_map)

        # Generate QA
        dspec_sky.make_model_qa(skymodel.wave, skyflux, skyvar, 
            wave, sky_truth, outfil='QA_sky_deconvovle_mean_'+camera+'.pdf')
        # 
        #xdb.set_trace()

def qa_mean_spec(wave, sky_model, sky_sub, sky_var, true_wave, true_sky, 
    frac_res=False, outfil=None):
    """
    Generate QA plots and files
    Parameters:
    true_wave, true_sky:  ndarrays
      Photons/s from simspec file
    """
    # Mean spectrum
    if outfil is None:
        outfil = 'tmp_qa_mean_sky.pdf'

    # Resample
    if true_wave is not None:
        dw = wave[1]-wave[0]
        ww = wave[0] + np.arange(len(wave))*dw
        true_flux = desi_interp.resample_flux(ww, true_wave, true_sky)
    else:
        true_flux = true_sky            
    #import pdb
    #pdb.set_trace()

    # Scale
    scl = np.median(sky_model/true_flux)
    print('scale = {:g}'.format(scl))

    # Error
    if sky_var is not None:
        sky_sig = np.sqrt(sky_var)

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,1)

    xmin,xmax = np.min(wave), np.max(wave)
    # Simple spectrum plot
    ax_flux = plt.subplot(gs[0])
    ax_flux.plot(wave, sky_model, label='DESI')
    #ax_flux.plot(wave, sky_sig, label='Model Error')
    ax_flux.plot(wave,true_flux*scl, label='Truth')
    ax_flux.get_xaxis().set_ticks([]) # Suppress labeling
    ax_flux.set_ylabel('Counts')
    ax_flux.set_xlim(xmin,xmax)
    ax_flux.text(0.5, 0.85, 'Sky Meanspec',
        transform=ax_flux.transAxes, ha='center')
    ax_flux.set_yscale("log", nonposy='clip')

    # Legend
    legend = ax_flux.legend(loc='upper left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Residuals
    scatt_sz = 0.5
    ax_res = plt.subplot(gs[1])
    ax_res.get_xaxis().set_ticks([]) # Suppress labeling
    res = (sky_model - (true_flux*scl))/(true_flux*scl)
    rms = np.sqrt(np.sum(res**2)/len(res))
    ax_res.set_ylim(-3.*rms, 3.*rms)
    #ax_res.set_ylim(-2, 2)
    ax_res.set_ylabel('Frac Res')
    # Error
    #ax_res.plot(true_wave, 2.*ms_sig/sky_model, color='red')
    ax_res.scatter(wave,res, marker='o',s=scatt_sz)
    ax_res.plot([xmin,xmax], [0.,0], 'g-')
    ax_res.set_xlim(xmin,xmax)

    # Relative to error
    if sky_var is not None:
        ax_sig = plt.subplot(gs[2])
        ax_sig.set_xlabel('Wavelength')
        sig_res = (sky_model - (true_flux*scl))/sky_sig
        ax_sig.scatter(wave, sig_res, marker='o',s=scatt_sz)
        ax_sig.set_ylabel(r'Res $\delta/\sigma$')
        ax_sig.set_ylim(-5., 5.)
        ax_sig.plot([xmin,xmax], [0.,0], 'g-')
        ax_sig.set_xlim(xmin,xmax)

    # Sky sub
    if sky_sub is not None:
        ax_sub = plt.subplot(gs[2])
        ax_sub.set_xlabel('Wavelength')
        sub_res = sky_sub
        ax_sub.scatter(wave, sub_res/(true_flux*scl), marker='o',s=scatt_sz)
        ax_sub.set_ylabel('Frac Sub Res')
        ax_sub.set_ylim(-0.1, 0.1)
        ax_sub.plot([xmin,xmax], [0.,0], 'g-')
        ax_sub.set_xlim(xmin,xmax)

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)

def qa_fiber_stats(frame, skyfibers, outfil=None):
    """
    Generate QA plot of fiber stats for individual sky-subtracted Sky fibers
    Parameters:
    frame
    skyfibers
    """
    from scipy.stats import norm, chi2

    # Consider going to additional pages
    nsky = len(skyfibers)
    xmin, xmax = np.min(skyfibers)-5, np.max(skyfibers)+5

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(1,2)

    # Do stats
    means = np.zeros(nsky)
    medians = np.zeros(nsky)
    red_chi2 = np.zeros(nsky)
    for kk,skyfiber in enumerate(skyfibers):
        # Histogram the sigma (should avoid bad ivar pixels)
        msk = np.where(frame.ivar[skyfiber,:] > 0.)[0]
        nsig = frame.flux[skyfiber,msk] * np.sqrt(frame.ivar[skyfiber,msk])
        red_chi2[kk] = np.sum( nsig**2) / len(msk)
        means[kk] = np.mean(nsig)
        medians[kk] = np.median(nsig)
        #print('Mean={:g}, chi^2={:g} for fiber={:d}'.format(
        #    np.mean(nsig),red_chi2,kk))

    # Global
    jj=0
    ax= plt.subplot(gs[jj])

    xscatt = np.arange(nsky)
    ax.scatter(skyfibers, medians, marker='o', s=1, label=r'Median $(\delta/\sigma)$')
    ax.scatter(skyfibers, red_chi2-1., marker='s', edgecolor='blue',
        facecolor='none',s=3, label=r'$\chi^2_\nu-1$')

    # Axes
    #ax_flux.set_ylabel('N')
    #ax.set_xlim(xmin,xmax)
    #lbls = ['{:d}'.format(skyfiber) for skyfiber in skyfibers]
    #ax.get_xaxis().set_ticks(skyfibers,lbls)
    #ax.set_ylim(-0.1, 0.2)
    ax.plot([xmin,xmax], [0.,0], 'g--')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Fiber')

    # Legend
    legend = ax.legend(loc='lower left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')
    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfil is not None:
        plt.savefig(outfil)

def qa_fiber_chi(frame, skyfibers, ncol=10, outfil=None):
    """
    Generate QA plots for individual sky-subtracted Sky fibers
    Parameters:
    frame
    skyfibers
    ncol: int, optional
      Number of columns in plot
    """
    from scipy.stats import norm, chi2

    # Mean spectrum

    # Consider going to additional pages
    nsky = len(skyfibers)
    nrow = nsky // ncol + ((nsky%ncol) > 0)

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(nrow,ncol)

    xmin,xmax = -5., 5. # sigma
    xchi_min, xchi_max = norm.cdf(xmin), norm.cdf(xmax)
    xchi = np.linspace(norm.ppf(xchi_min),norm.ppf(xchi_max),100)
    binsz = 0.5

    for kk,skyfiber in enumerate(skyfibers):
        # Histogram the sigma (should avoid bad ivar pixels)
        msk = np.where(frame.ivar[skyfiber,:] > 0.)[0]
        nsig = frame.flux[skyfiber,msk] * np.sqrt(frame.ivar[skyfiber,msk])
        red_chi2 = np.sum( nsig**2) / len(msk)
        print('Mean={:g}, chi^2={:g} for fiber={:d}'.format(
            np.mean(nsig),red_chi2,kk))

        # Simple histogram
        i0 = int( np.min(nsig) / binsz) - 1
        i1 = int( np.max(nsig) / binsz) + 1
        rng = tuple( binsz*np.array([i0,i1]) )
        nbin = i1-i0
        # Histogram
        hist, edges = np.histogram(nsig, range=rng, bins=nbin)
        xhist = (edges[1:] + edges[:-1])/2.
        # Plot histogram
        ax= plt.subplot(gs[kk])
        #ax.bar(edges[:-1], hist, width=binsz)
        ax.hist(xhist, bins=edges, weights=hist, histtype='step')

        # P(Chi) 
        ychi = binsz*len(msk)*norm.pdf(xchi)
        ax.plot(xchi, ychi, color='red')

        # Axes
        #ax_flux.set_ylabel('N')
        ax.set_xlim(xmin,xmax)
        #ax.text(0.5, 0.85, '{:d}'.format(kk), transform=ax.transAxes, ha='center')
        ax.get_yaxis().set_ticks([]) # Suppress labeling
        ax.get_xaxis().set_ticks([]) # Suppress labeling
        if kk >= (ncol-2)*nrow:
            ax.set_xlabel(r'$n\sigma$')

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfil is not None:
        plt.savefig(outfil)

# ##################################################
# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg_tst = 0 
    #flg_tst += 2**0  # Deconvolved mean [DEPRECATED]
    flg_tst += 2**1  # Sky fiber mean 

    if (flg_tst % 2**1) >= 2**0:
        path = '/Users/xavier/DESI/TST/20150211/' 
        tst_deconvolve_mean_sky('/Users/xavier/DESI/TST/20150211/simspec-00000002.fits',
            '0-00000002.fits', 'fibermap-00000002.fits',
            path=path)

    if (flg_tst % 2**2) >= 2**1:
        path = '/Users/xavier/DESI/TST/20150211/' 
        tst_meansky_fibers('/Users/xavier/DESI/TST/20150211/simspec-00000002.fits',
            '0-00000002.fits', '0-00000001.fits', path=path)
