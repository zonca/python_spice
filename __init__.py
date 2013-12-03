import numpy as np
import pyfits
import subprocess
import sys
import pandas as pd
import os
import json
import healpy as hp
from glob import glob

def to_json(cl, name):
    cl_json = {}
    for comp in cl.keys():
        cl_json[comp] = []
        for ell, value in cl[comp].iteritems():
            cl_json[comp].append({"x":ell, "y":value})
    with open(name + ".json", "w") as f:
        json.dump(cl_json, f)

def default_parameters():
    return {
        "apodizesigma":"NO",
        "apodizetype":"NO",
        "beam":"NO",
        "beam_file":"beam.csv",
        "beam2":"NO",
        "beam_file2":"beam.csv",
        "corfile":"NO",
        "clfile":"spiceTEMP_spice_cl.fits",
        "cl_outmap_file":"NO",
        "cl_inmap_file":"NO",
        "cl_outmask_file":"NO",
        "cl_inmask_file":"NO",
        "covfileout":"NO",
        "decouple":"NO",
        "dry":"YES",
        "extramapfile":"YES",
        "extramapfile2":"YES",
        "fits_out":"YES",
        "kernelsfileout":"NO",
        "mapfile":"YES",
        "mapfile2":"NO",
        "maskfile":"total_mask_1024.fits",
        "maskfile2":"total_mask_1024.fits",
        "maskfilep":"total_mask_1024.fits",
        "maskfilep2":"total_mask_1024.fits",
        "nlmax":"1024",
        "normfac":"NO",
        "npairsthreshold":"NO",
        "noisecorfile":"NO",
        "noiseclfile":"NO",
        "overwrite":"YES",
        "polarization":"YES",
        "pixelfile":"YES",
        "subav":"NO",
        "subdipole":"NO",
        "symmetric_cl":"NO",
        "tf_file":"NO",
        "thetamax":"NO",
        "verbosity":"1",
        "weightfile":"NO",
        "weightfilep":"YES",
        "weightfile2":"NO",
        "weightfilep2":"YES",
        "weightpower":"1.00000000000000",
        "weightpower2":"1.00000000000000",
        "weightpowerp":"1.00000000000000",
        "weightpowerp2":"1.00000000000000",
        "windowfilein":"NO",
        "windowfileout":"NO"
}

def write_params(params, filename="spiceTEMP_spice_params.txt"):
    with open(filename, "w") as f:
        for k in sorted(params.keys()):
            f.write("%s = %s\n" % (k, params[k]))

def read_cl(filename="./spiceTEMP_spice_cl.fits"):
    cl_file = pyfits.open(filename)
    cl_file[1].verify("fix")
    cl = pd.DataFrame(np.array(cl_file[1].data))
    # convert from string (?!) to float
    cl[cl.keys()] = cl[cl.keys()].astype(np.float32)
    cl.index.name = "ell"
    return cl

def compute_llp1(ell):
    return ell * (ell+1) / 2. / np.pi

def compute_power(ell):
    return (2*ell+1) / 4. / np.pi # power

def weighted_average(g):
    weights = compute_power(g.index)
    return g.mul(weights, axis="index").sum()/weights.sum() 

def bin_cl(cl):
    ctp_binning = pd.read_csv("planck_ctp_bin.csv")
    #ctp_binning = ctp_binning[0:ctp_binning["last l"].searchsorted(1024)+1]
    bins = pd.cut(cl.index, bins=np.concatenate([[1], ctp_binning["last l"]]))
    cl["ell"] = np.float64(cl.index)
    binned_cl = cl.groupby(bins).apply(weighted_average)
    binned_cl = binned_cl.set_index("ell")
    return binned_cl

def spice(bin=True, norm=True, **kwargs):
    """Run spice (needs to be in PATH)

    Parameters
    ----------
    bin : bool
        apply C_ell binning (planck_ctp)
    norm : bool
        return C_ell * ell(ell+1)/2pi [muK^2]
    other_arguments:
        all spice arguments, see `default_params` in the module source
        for example:
        mapfile is the input map for spectra, assumed to be in [K]
        mapfile2 is the second map for cross-spectra

    Returns
    -------
    cl : pd.Series
        pandas Series of C_ell[K^2] (if norm is False) or C_ell * ell(ell+1)/2pi [muK^2]
        (if norm is True)
    """
    params = default_parameters()

    # write arrays maps to disk
    for key in ["mapfile", "mapfile2"]:
        if kwargs.has_key(key):
            if not isinstance(kwargs[key], str):
                temp_map_file = "spiceTEMP_" + key + ".fits"
                hp.write_map(temp_map_file, kwargs[key])
                kwargs[key] = temp_map_file

    params.update(kwargs)
    write_params(params)
    try:
        subprocess.check_output(["spice", "-optinfile", "./spiceTEMP_spice_params.txt"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error in spice")
        print(e.output)
        sys.exit(1)
    cl = read_cl()
    for f in glob("spiceTEMP*"):
        os.remove(f)
    lmax = cl.index.max()
    if bin:
        cl = bin_cl(cl)
    if norm:
        cl = cl.mul(1e12 * compute_llp1(np.array(cl.index)), axis="index")
    return cl[cl.index < lmax]
