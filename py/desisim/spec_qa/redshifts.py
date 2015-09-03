"""
desisim.spec_qa.redshifts
============

Utility functions to do simple QA on the Redshifts
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

def load_z(fibermap_files=None, zbest_files=None, path=None):
    '''Load input and output redshift values for a set of exposures
    '''
    # imports

    # Init
    simspec_path = '/Users/xavier/DESI/TST/20150211/'
    fibermap_path = '/Users/xavier/DESI/TST/20150211/'
    zbest_path = '/Users/xavier/DESI/TST/20150211/bricks/'
    if fibermap_files is None:
        fibermap_files = glob.glob(fibermap_path+'fibermap-*')

    # Load up fibermap and simspec tables
    fbm_tabs = []
    sps_tabs = []
    for fibermap_file in fibermap_files:
        fbm_hdu = fits.open(fibermap_file)
        print('Reading: {:s}'.format(fibermap_file))
        # Load simspec
        ffil = fibermap_file[fibermap_file.rfind('fiber'):]
        simspec_fil = simspec_path+ffil.replace('fibermap','simspec')
        sps_hdu = fits.open(simspec_fil)
        # Make Tables
        assert fbm_hdu[1].name == 'FIBERMAP'
        fbm_tabs.append(Table(fbm_hdu[1].data))
        assert sps_hdu[2].name == 'METADATA'
        sps_tabs.append(Table(sps_hdu[2].data))
    # Stack
    fbm_tab = vstack(fbm_tabs)
    sps_tab = vstack(sps_tabs)
    del fbm_tabs, sps_tabs

    # Drop to unique
    univ, uni_idx = np.unique(np.array(fbm_tab['TARGETID']),return_index=True)
    fbm_tab = fbm_tab[uni_idx]
    sps_tab = sps_tab[uni_idx]

    # Combine + Sort
    sim_tab = hstack([fbm_tab,sps_tab],join_type='exact')
    sim_tab.sort('TARGETID')
    nsim = len(sim_tab)

    # Load up zbest files
    if zbest_files is None:
        zbest_files = glob.glob(zbest_path+'zbest-*')
    zb_tabs = []
    for zbest_file in zbest_files:
        zb_hdu = fits.open(zbest_file)
        zb_tabs.append(Table(zb_hdu[1].data))
    # Stack
    zb_tab = vstack(zb_tabs)
    univ, uni_idx = np.unique(np.array(zb_tab['TARGETID']),return_index=True)
    zb_tab = zb_tab[uni_idx]

    # Match up
    sim_id = np.array(sim_tab['TARGETID'])
    z_id = np.array(zb_tab['TARGETID'])
    inz = np.in1d(z_id,sim_id,assume_unique=True)
    ins = np.in1d(sim_id,z_id,assume_unique=True)

    z_idx = np.arange(z_id.shape[0])[inz]
    sim_idx = np.arange(sim_id.shape[0])[ins]
    assert np.array_equal(sim_id[sim_idx],z_id[z_idx])

    # Fill up
    ztags = ['Z','ZERR','ZWARN','TYPE']
    new_tags = ['REDM_Z','REDM_ZERR','REDM_ZWARN','REDM_TYPE']
    new_clms = []
    mask = np.array([True]*nsim)
    mask[sim_idx] = False
    for kk,ztag in enumerate(ztags):
        # Generate a MaskedColumn
        new_clm  = MaskedColumn([zb_tab[ztag][z_idx[0]]]*nsim,
            name=new_tags[kk], mask=mask)
        # Fill
        new_clm[sim_idx] = zb_tab[ztag][z_idx]
        # Append
        new_clms.append(new_clm)
    # Add columns
    sim_tab.add_columns(new_clms)

    # Return
    return sim_tab # Masked Table

def qa_z_summ(sim_tab, outfil=None):
    """
    Generate QA plot of 
    Parameters:
    frame
    skyfibers
    """
    from scipy.stats import norm, chi2

    if outfil is None:
        outfil = 'qa_z_summ.pdf'

    # Plot
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(1,2)

    # Cut table
    cut_tab = sim_tab[np.where(sim_tab['REDM_Z'].mask == False)[0]]

    otypes = ['ELG','LRG','QSO']
    sty_otype = dict(ELG={'color':'green'},
        LRG={'color':'red'},
        QSO={'color':'blue'})

    # Global
    jj=0
    ax= plt.subplot(gs[jj])

    for otype in otypes: 
        # Grab
        gd_z = np.where(cut_tab['OBJTYPE_1']==otype)[0]
        ax.scatter(cut_tab['REDSHIFT'][gd_z], cut_tab['REDM_Z'][gd_z], 
            marker='o', s=1, label=otype, color=sty_otype[otype]['color'])

    #ax.set_xlim(xmin, xmax)
    ax.set_ylabel(r'$z_{\rm red}$')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_xlim(0.,4)

    # Zoom
    jj=1
    ax= plt.subplot(gs[jj])

    for otype in otypes: 
        # Grab
        gd_z = np.where(cut_tab['OBJTYPE_1']==otype)[0]
        # Stat
        dz = ((cut_tab['REDM_Z'][gd_z]-cut_tab['REDSHIFT'][gd_z])/ 
            (1+cut_tab['REDSHIFT'][gd_z]))

        ax.scatter(cut_tab['REDSHIFT'][gd_z], dz, marker='o', 
            s=1, label=otype, color=sty_otype[otype]['color'])

    #ax.set_xlim(xmin, xmax)
    ax.set_ylabel(r'$(z_{\rm red}-z_{\rm true}) / (1+z)$')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_xlim(0.,4)
    deltaz = 0.002
    ax.set_ylim(-deltaz,deltaz)

    # Legend
    legend = ax.legend(loc='lower left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')
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
    #flg_tst += 2**0  # load redshifts
    flg_tst += 2**1  # simple redshift stats

    if (flg_tst % 2**1) >= 2**0:
        sim_tab=load_z()

    if (flg_tst % 2**2) >= 2**1:
        sim_tab=load_z()
        qa_z_summ(sim_tab)

