"""
I/O routines
"""

import os
import time

from astropy.io import fits
import numpy as np
import multiprocessing

from desispec.interpolation import resample_flux
from desispec.io.util import write_bintable, header2wave
import desispec.io
import desimodel.io

#-------------------------------------------------------------------------
#- simspec

def write_simspec(meta, truth, expid, night, header=None, outfile=None):
    """
    Write $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
    
    Args:
        meta : metadata table to write to "METADATA" HDU
        truth : dictionary with keys:
            FLUX - 2D array [nspec, nwave] in erg/s/cm2/A
            WAVE - 1D array of vacuum wavelengths [Angstroms]
            SKYFLUX - array of sky flux [erg/s/cm2/A/arcsec],
                      either 1D [nwave] or 2D [nspec, nwave]
            PHOT_{B,R,Z} - 2D array [nspec, nwave] of object photons/bin
            SKYPHOT_{B,R,Z} - 1D or 2D array of sky photons/bin
        expid : integer exposure ID
        night : string YEARMMDD
        header : optional dictionary of header items to add to output
        outfile : optional filename to write (otherwise auto-derived)
        
    Returns:
        full file path of output file written
        
    """
    #- Where should this go?
    if outfile is None:
        outdir = simdir(night, mkdir=True)      
        outfile = '{}/simspec-{:08d}.fits'.format(outdir, expid)

    #- Primary HDU is just a header from the input
    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(None, header=desispec.io.util.fitsheader(header)))

    #- Object flux HDU (might not exist, e.g. for an arc)
    if 'FLUX' in truth:
        x = fits.ImageHDU(truth['WAVE'], name='WAVE')
        x.header['BUNIT']  = ('Angstrom', 'Wavelength units')
        x.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hx.append(x)

        x = fits.ImageHDU(truth['FLUX'].astype(np.float32), name='FLUX')
        x.header['BUNIT'] = '1e-17 erg/s/cm2/A'
        hx.append(x)
    
    #- Sky flux HDU
    if 'SKYFLUX' in truth:
        x = fits.ImageHDU(truth['SKYFLUX'].astype(np.float32), name='SKYFLUX')
        x.header['BUNIT'] = '1e-17 erg/s/cm2/A/arcsec2'
        hx.append(x)
    
    #- Write object photon and sky photons for each channel
    for channel in ['B', 'R', 'Z']:
        x = fits.ImageHDU(truth['WAVE_'+channel], name='WAVE_'+channel)
        x.header['BUNIT']  = ('Angstrom', 'Wavelength units')
        x.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hx.append(x)

        extname = 'PHOT_'+channel
        x = fits.ImageHDU(truth[extname].astype(np.float32), name=extname)
        x.header['EXTNAME'] = (extname, channel+' channel object photons per bin')
        hx.append(x)

        extname = 'SKYPHOT_'+channel
        if extname in truth:
            x = fits.ImageHDU(truth[extname].astype(np.float32), name=extname)
            x.header['EXTNAME'] = (extname, channel+' channel sky photons per bin')
            hx.append(x)

    #- Write the file
    hx.writeto(outfile, clobber=True)

    #- Add Metadata table HDU; use write_bintable to get units and comments
    if meta is not None:
        comments = dict(
            OBJTYPE     = 'Object type (ELG, LRG, QSO, STD, STAR)',
            REDSHIFT    = 'true object redshift',
            TEMPLATEID  = 'input template ID',
            O2FLUX      = '[OII] flux [erg/s/cm2]',
        )
    
        units = dict(
            # OBJTYPE     = 'Object type (ELG, LRG, QSO, STD, STAR)',
            # REDSHIFT    = 'true object redshift',
            # TEMPLATEID  = 'input template ID',
            O2FLUX      = 'erg/s/cm2',
        )
    
        write_bintable(outfile, meta, header=None, extname="METADATA",
            comments=comments, units=units)

    return outfile

class SimSpec(object):
    """Lightweight wrapper object for simspec data"""
    def __init__(self, flavor, wave, phot, flux=None, skyflux=None,
                 skyphot=None, metadata=None, header=None):
        """
        Args:
            flavor : 'arc', 'flat', or 'science'
            wave : dictionary with per-channel wavelength grids, keyed by
                'b', 'r', 'z'.  Optionally also has 'brz' key for channel
                independent wavelength grid
            phot : dictinoary with per-channel photon counts per bin

        Optional:
            flux : channel-independent flux [erg/s/cm^2/A]
            skyflux : channel-indepenent sky flux [erg/s/cm^2/A/arcsec^2]
            skyphot : dictionary with per-channel sky photon counts per bin
            metadata : table of metadata information about these spectra
            header : FITS header from HDU0

        notes:
          * input arguments become attributes
          * wave[channel] is the wavelength grid for phot[channel] and
                skyphot[channel] where channel = 'b', 'r', or 'z'
          * wave['brz'] is the wavelength grid for flux and skyflux
        """
        assert flavor in ('arc', 'flat', 'science')
        for channel in ('b', 'r', 'z'):
            assert wave[channel].ndim == 1
            assert phot[channel].ndim == 2
            assert wave[channel].shape[0] == phot[channel].shape[1]

        assert phot['b'].shape[0] == phot['r'].shape[0] == phot['z'].shape[0]

        self.flavor = flavor
        self.nspec = phot['b'].shape[0]
        self.wave = wave
        self.phot = phot

        #- Optional items; may be None
        self.skyphot = skyphot
        self.flux = flux
        self.skyflux = skyflux
        self.metadata = metadata
        self.header = header

def read_simspec(filename):
    """
    Read simspec data from filename and return SimSpec object
    """

    fx = fits.open(filename)
    hdr = fx[0].header
    flavor = hdr['FLAVOR']

    #- All flavors have photons
    wave = dict()
    phot = dict()
    for channel in ('b', 'r', 'z'):
        wave[channel] = fx['WAVE_'+channel.upper()].data
        phot[channel] = fx['PHOT_'+channel.upper()].data

    if flavor == 'arc':
        fx.close()
        return SimSpec(flavor, wave, phot, header=hdr)

    elif flavor == 'flat':
        wave['brz'] = fx['WAVE'].data
        flux = fx['FLUX'].data
        fx.close()
        return SimSpec(flavor, wave, phot, flux=flux, header=hdr)

    elif flavor == 'science':
        wave['brz'] = fx['WAVE'].data
        flux = fx['FLUX'].data
        metadata = fx['METADATA'].data
        skyflux = fx['SKYFLUX'].data
        skyphot = dict()
        for channel in ('b', 'r', 'z'):
            extname = 'SKYPHOT_'+channel.upper()
            skyphot[channel] = fx[extname].data

        fx.close()
        return SimSpec(flavor, wave, phot, flux=flux, skyflux=skyflux,
            skyphot=skyphot, metadata=metadata, header=hdr)

    else:
        raise ValueError('unknown flavor '+flavor)

    
    
#- TODO: this is more than just I/O.  Refactor.
def write_simpix(img, camera, night, expid, header):
    """
    Add noise to input image and write output simpix and pix files.
    
    Args:
        img : 2D noiseless image array
        camera : e.g. b0, r1, z9
        flavor : arc or flat
        night  : YEARMMDD string
        expid  : integer exposure id
        header : dict-like object that should include FLAVOR and EXPTIME,
            e.g. from HDU0 FITS header of input simspec file
        
    Writes to $DESI_SPECTRO_SIM/$PIXPROD/{night}/
        simpix-{camera}-{expid}.fits
        pix-{camera}-{expid}.fits
        
    Returns:
        filepath to pix*.fits file that was written
    """

    outdir = simdir(night, mkdir=True)
    params = desimodel.io.load_desiparams()
    channel = camera[0].lower()

    #- Add noise, generate inverse variance and mask
    rdnoise = params['ccd'][channel]['readnoise']
    pix = np.random.poisson(img) + np.random.normal(scale=rdnoise, size=img.shape)
    ivar = 1.0/(pix.clip(0) + rdnoise**2)
    mask = np.zeros(img.shape, dtype=np.int32)

    #-----
    #- Write noiseless image to simpix file
    simpixfile = '{}/simpix-{}-{:08d}.fits'.format(outdir, camera, expid)

    hdu = fits.PrimaryHDU(img, header=header)
    hdu.header['VSPECTER'] = ('0.0.0', 'TODO: Specter version')
    fits.writeto(simpixfile, hdu.data, header=hdu.header, clobber=True)

    #- Add x y trace locations from PSF
    psffile = '{}/data/specpsf/psf-{}.fits'.format(os.getenv('DESIMODEL'), channel)
    psfxy = fits.open(psffile)
    fits.append(simpixfile, psfxy['XCOEFF'].data, header=psfxy['XCOEFF'].header)
    fits.append(simpixfile, psfxy['YCOEFF'].data, header=psfxy['YCOEFF'].header)

    #-----
    #- Write simulated raw data to pix file

    #- Primary HDU: noisy image
    outfile = '{}/pix-{}-{:08d}.fits'.format(outdir, camera, expid)
    hdulist = fits.HDUList()
    if 'EXTNAME' in header:
        del header['EXTNAME']
    hdu = fits.PrimaryHDU(pix, header=header)
    hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )
    hdu.header.append( ('RDNOISE', rdnoise, 'Read noise [electrons]'))
    hdulist.append(hdu)

    #- IVAR: Inverse variance (IVAR)
    hdu = fits.ImageHDU(ivar, name='IVAR')
    hdu.header.append(('RDNOISE', rdnoise, 'Read noise [electrons]'))
    hdulist.append(hdu)

    #- MASK: currently just zeros
    hdu = fits.CompImageHDU(mask, name='MASK')
    hdulist.append(hdu)

    hdulist.writeto(outfile, clobber=True)

    return outfile


#-------------------------------------------------------------------------
#- desimodel

def get_tile_radec(tileid):
    """
    Return (ra, dec) in degrees for the requested tileid.
    
    If tileid is not in DESI, return (0.0, 0.0)
    TODO: should it raise and exception instead?
    """
    tiles = desimodel.io.load_tiles()
    if tileid in tiles['TILEID']:
        i = np.where(tiles['TILEID'] == tileid)[0][0]
        return tiles[i]['RA'], tiles[i]['DEC']
    else:
        return (0.0, 0.0)   

#-------------------------------------------------------------------------
#- spectral templates

#- Utility function to wrap resample_flux for multiprocessing map
def _resample_flux(args):
    return resample_flux(*args)

def read_templates(wave, objtype, nspec=None, randseed=1, infile=None):
    """
    Returns n templates of type objtype sampled at wave
    
    Inputs:
      - wave : array of wavelengths to sample
      - objtype : 'ELG', 'LRG', 'QSO', 'STD', or 'STAR'
      - nspec : number of templates to return
      - infile : (optional) input template file (see below)
    
    Returns flux[n, len(wave)], meta[n]

    where flux is in units of 1e-17 erg/s/cm2/A/[arcsec^2] and    
    meta is a metadata table from the input template file
    with redshift, mags, etc.
    
    If infile is None, then $DESI_{objtype}_TEMPLATES must be set, pointing to
    a file that has the observer frame flux in HDU 0 and a metadata table for
    these objects in HDU 1. This code randomly samples n spectra from that file.
    
    TO DO: add a setable randseed for random reproducibility.
    """
    if infile is None:
        key = 'DESI_'+objtype.upper()+'_TEMPLATES'
        if key not in os.environ:
            raise ValueError("ERROR: $"+key+" not set; can't find "+objtype+" templates")
        
        infile = os.getenv(key)

    hdr = fits.getheader(infile)
    flux = fits.getdata(infile, 0)
    meta = fits.getdata(infile, 1).view(np.recarray)
    ww = 10**(hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1'])

    #- Check flux units
    fluxunits = hdr['BUNIT']
    if not fluxunits.startswith('1e-17 erg'):
        if fluxunits.startswith('erg'):
            flux *= 1e17
        else:
            #- check for '1e-16 erg/s/cm2/A' style units
            scale, units = fluxunits.split()
            assert units.startswith('erg')
            scale = float(scale)
            flux *= (scale*1e17)

    ntemplates = flux.shape[0]
    randindex = np.arange(ntemplates)
    np.random.shuffle(randindex)
    
    if nspec is None:
        nspec = flux.shape[0]
    
    #- Serial version
    # outflux = np.zeros([n, len(wave)])
    # outmeta = np.empty(n, dtype=meta.dtype)
    # for i in range(n):
    #     j = randindex[i%ntemplates]
    #     if 'Z' in meta:
    #         z = meta['Z'][j]
    #     else:
    #         z = 0.0
    #     if objtype == 'QSO':
    #         outflux[i] = resample_flux(wave, ww, flux[j])
    #     else:
    #         outflux[i] = resample_flux(wave, ww*(1+z), flux[j])
    #     outmeta[i] = meta[j]
        
    #- Multiprocessing version
    #- Assemble list of args to pass to multiprocesssing map
    args = list()
    outmeta = np.empty(nspec, dtype=meta.dtype)
    for i in range(nspec):
        j = randindex[i%ntemplates]
        outmeta[i] = meta[j]
        if 'Z' in meta.dtype.names:
            z = meta['Z'][j]
        else:
            z = 0.0
    
        #- ELG, LRG require shifting wave by (1+z); QSOs don't
        if objtype == 'QSO':
            args.append( (wave, ww, flux[j]) )
        else:
            args.append( (wave, ww*(1+z), flux[j]) )
        
    ncpu = multiprocessing.cpu_count() // 2   #- avoid hyperthreading
    pool = multiprocessing.Pool(ncpu)
    outflux = pool.map(_resample_flux, args)        
        
    return outflux, outmeta
    

#-------------------------------------------------------------------------
#- Utility functions

def simdir(night='', mkdir=False):
    """
    Return $DESI_SPECTRO_SIM/$PIXPROD/{night}
    If mkdir is True, create directory if needed
    """
    dirname = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), night)        
    if mkdir and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    return dirname
    
def _parse_filename(filename):
    """
    Parse filename and return (prefix, expid) or (prefix, camera, expid)
    """
    base = os.path.basename(os.path.splitext(filename)[0])
    x = base.split('-')
    if len(x) == 2:
        return x[0], None, int(x[1])
    elif len(x) == 3:
        return x[0], x[1].lower(), int(x[2])
        
    

