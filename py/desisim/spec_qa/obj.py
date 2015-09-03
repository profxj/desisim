"""
desisim.spec_qa.obj
============

Utility functions to do simple QA on the objects
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys, os, pdb, glob

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn

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

from xastropy.xutils import xdebug as xdb

def compare_spec(fibermap_file, TARGETID):
    '''Compare input (simulated) spectrum with DESI output
    '''
    # imports

    # Load fiber file
    fmap = Table.read(fibermap_file)
    idx = np.where(fmap['TARGETID']==TARGETID)[0]
    if len(idx) == 0:
        print('TARGETID={:d} not in {:s}'.format(TARGETID,fibermap_file))
        return
    idx = idx[0]
    spectroid = fmap['SPECTROID'][idx]
    # Load simulated spectrum
    simspec_fil = fibermap_file.replace('fibermap','simspec')
    sim_hdu = fits.open(simspec_fil)
    hdr = sim_hdu[0].header # This will change
    sim_wave = hdr['CRVAL1'] + (hdr['CDELT1'])*np.arange(hdr['NAXIS1'])
    sim_flux = sim_hdu[0].data[:,idx]

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,1)

    cameras = ['B','Z','R']
    for camera in cameras:
        # Grab cframe
        i0 = fibermap_file.rfind('/')
        i1 = fibermap_file.rfind('-')
        cframe_file = fibermap_file[0:i0]+'/cframe-'+camera.lower()+'{:d}'.format(spectroid)+fibermap_file[i1:]
        cframe = desi_io_frame.read_frame(cframe_file)
        #

        ax= plt.subplot(gs[jj])

        #ax.set_xlim(xmin, xmax)
        ax.set_ylabel('Flux')
        ax.set_xlabel('Wavelength')
        ax.set_xlim(0.,4)

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfil is not None:
        plt.savefig(outfil)

def qa_z_elg(sim_tab, outfil=None):
    """
    Generate QA plot for ELGs
    Parameters:
    sim_tab
    """
    from scipy.stats import norm, chi2

    if outfil is None:
        outfil = 'qa_z_elg.pdf'

    deltaz = 0.00035

    # Plot
    fig = plt.figure(figsize=(8, 6.0))
    gs = gridspec.GridSpec(2,2)

    # Cut table
    cut_tab = sim_tab[np.where(sim_tab['REDM_Z'].mask == False)[0]]
    cut_tab = cut_tab[np.where(cut_tab['OBJTYPE_1'] == 'ELG')[0]]

    sty_otype = dict(ELG={'color':'green'},
        LRG={'color':'red'},
        QSO={'color':'blue'})

    # Offset
    for kk in range(2):
        # y-axis
        if kk == 0:
            dz = ((cut_tab['REDM_Z']-cut_tab['REDSHIFT'])/ 
                (1+cut_tab['REDSHIFT']))
            ylbl = (r'$(z_{\rm red}-z_{\rm true}) / (1+z)$')
            ylim = deltaz
        elif kk == 1:
            dz = ((cut_tab['REDM_Z']-cut_tab['REDSHIFT'])/ 
                (cut_tab['REDM_ZERR']))
            ylbl = (r'$(z_{\rm red}-z_{\rm true}) / \sigma(z)$')
            ylim = 3.
        # x-axis
        for jj in range(2):
            ax= plt.subplot(gs[kk,jj])
            if jj == 0:
                lbl = r'$z_{\rm true}$'
                xval = cut_tab['REDSHIFT']
                xmin,xmax=0.6,1.65
            elif jj == 1:
                lbl = r'[OII] Flux ($10^{-16}$)'
                xval = cut_tab['O2FLUX']*1e16
                xmin,xmax=0.3,20
                ax.set_xscale("log", nonposy='clip')
            # x labels
            ax.set_ylabel(ylbl)
            ax.set_xlabel(lbl)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(-ylim, ylim)

            # Points
            ax.scatter(xval, dz, marker='o', 
                s=1, label='ELG', color=sty_otype['ELG']['color'])

    # Legend
    #legend = ax.legend(loc='lower left', borderpad=0.3,
    #                    handletextpad=0.3, fontsize='small')
    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.0)
    if outfil is not None:
        plt.savefig(outfil)

# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg_tst = 0 
    flg_tst += 2**0  # Compare DESI with input spectrum
    #flg_tst += 2**1  # simple redshift stats
    #flg_tst += 2**2  # ELGs
    #flg_tst += 2**3  # Summary table

    if (flg_tst % 2**1) >= 2**0:
        fibermap_file = '/Users/xavier/DESI/TST/20150211/fibermap-00000002.fits'
        targ_id = 8218809192091191290
        compare_spec(fibermap_file,targ_id)

