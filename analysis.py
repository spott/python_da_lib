import numpy
import pandas
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from __future__ import print_function, division

#we are interested in the energy levels:

def nm_to_hartree( nm ):
    length = 5.29177206e-11
    alpha = 1./137.035999
    return 2. * numpy.pi * length / (alpha * nm * 1e-9) 

#prototype load:
def get_prototype( filename = "prototype.csv" ):
    prototype_f = file( filename )

    prototype = []

    for l in prototype_f:
        i = l.split(',')
        a = dict()
        a["n"] = int(i[0])
        a["j"] = float( int( i[1] ) ) /2.
        a["l"] = int(i[2])
        a["e"] = float(i[4])
        prototype.append(a)
    return prototype

def find_channel_closings(wavelength_nm=800, Ip_au=.82277, nmax=5):
    trycalc=1000
    omega_au=nm_to_hartree(wavelength_nm)
    atomic_unit_of_intensity=3.50944758E16
    return filter(lambda x:x>0.0,[4 * omega_au**2 * (n * omega_au - Ip_au) * atomic_unit_of_intensity for n in range(0,trycalc)])[:nmax]

def import_complex_vector( filename ):
    with open (filename, "rb") as f:
        npy = numpy.fromfile(f, numpy.complex128, -1)
        return npy

def import_vector( filename ):
    with open (filename, "rb") as f:
        npy = numpy.fromfile(f, numpy.float64, -1)
        return npy

def import_petsc_vec( filename ):
    with open(filename, "rb") as f:
        byte = f.read(8)
        size = struct.unpack('>ii',byte)[1]
        npy = numpy.fromfile(f, '>D', size)
        return npy

def import_k_spectrum( filename ):
    with open(filename, "rb") as f:
        byte = f.read(8)
        size = struct.unpack('ii',byte)
        npy = numpy.fromfile(f, 'd', -1)
        return npy.reshape(size[0], size[1])

def gen_x_y_matrices( kmax, dk, dtheta ):
    print(kmax)
    print(dk)
    print(dtheta)
    dim_x = int(kmax / dk);
    dim_y = int(numpy.pi / dtheta) + 1;
    X = numpy.ndarray((dim_x, dim_y))
    Y = numpy.ndarray((dim_x, dim_y))
    for i in range(dim_x):
        for j in range(dim_y):
            X[i][j] = (i+1) * dk * numpy.cos( dtheta * j)
            Y[i][j] = (i+1) * dk * numpy.sin( dtheta * j)
    return (X,Y)

def gen_k_spectrum_figure( spectrum_filename, kmax, out_filename, r = []):
    ks = import_k_spectrum( spectrum_filename )
    ma = numpy.max(ks)
    print(ks.shape)
    print(ma)
    (X,Y) = gen_x_y_matrices( kmax, kmax / float(ks.shape[0]), numpy.pi / float(ks.shape[1]) )
    if (len(r) == 0):
        r.append(numpy.log10(ma) - 4)
        r.append(numpy.log10(ma))

    plt.figure()
    plt.pcolormesh(X, Y, numpy.log10(numpy.abs(ks)), norm=mpl.colors.Normalize(vmin=r[0], vmax=r[1], clip=True))
    plt.savefig( out_filename )
    #plt.figure()
    #plt.imshow(numpy.log10(numpy.abs(ks)), norm=mpl.colors.Normalize(vmin=(numpy.log10(ma) - 4), vmax=(numpy.log10(ma)), clip=True), interpolation="bilinear")
    #plt.savefig( out_filename )


def indexed_wf( prototype, wf ):
    r = pandas.DataFrame( prototype )
    r['wf'] = pandas.Series( wf )
    return r

def grouped_by( index, indexedwf ):
    #get an absolute value for the wf, otherwise we can't aggregate:
    r = indexedwf.copy()
    sqabs = lambda x : abs(x)**2
    r["wf"] = r["wf"].map( sqabs )
    g = r.groupby(index)
    r = g.aggregate( pandas.np.sum )
    return pandas.DataFrame({"wf_abs": r['wf']})

def binfunc( e, bins ):
    for b in bins:
        if e < b[0]:
            return b[1]
        else:
            continue
    return bins[-1][1]

def binned_group_by( indexedwf, bin_size = .001 ):
    r = indexedwf.copy()
    r["wf"] = r["wf"].map( lambda x: abs(x)**2 )
    minimum = min(r["e"])
    maximum = max(r["e"])
    diff = maximum - minimum
    #create the bins:
    bins = [ ((i + 1) * bin_size + minimum, (i + .5) * bin_size + minimum ) for i in range(int ( diff / bin_size ) + 1) ]
    bf = lambda x: binfunc( x, bins )
    g = r.set_index('e').groupby( bf )
    r = g.aggregate( pandas.np.sum )
    return pandas.DataFrame({"wf_abs":r['wf']})

def wavefunctions(folder = "./"):
    filenames = os.listdir(folder)
    wf_filenames = 0
    for i in filenames:
        if (i[:2] == "wf"):
            wf_filenames += 1
    wfs = {}
    for i in range(wf_filenames-1):
        wfs[i] = import_petsc_vec(folder + "/wf_" + str(i) + ".dat" ) 
    wfs[wf_filenames] = import_petsc_vec( folder + "/wf_final.dat")
    return wfs

def wavefunctions_non_zero(folder = "./"):
    filenames = os.listdir(folder)
    wf_filenames = 0
    for i in filenames:
        if (i[:4] == "wf_p"):
            wf_filenames += 1
    wfs = {}
    for i in range(wf_filenames-1):
        wfs[i] = import_petsc_vec(folder + "/wf_p" + str(i) + "o4.dat" ) 
    return wfs



def probabilities(folder = "./"):
    return pandas.DataFrame(wavefunctions(folder)).apply( lambda x: abs(x)**2 )

def grouped_probabilities( index, prototype, wfs):
    r = wfs.copy()
    r[index] = pandas.DataFrame(prototype)[index]
    return r.groupby(index).aggregate(pandas.np.sum)

def ionized_probabilities( prototype, wfs, llc = .06):
    r = wfs.copy()
    r["e"] = pandas.DataFrame(prototype)["e"]
    g = r.set_index("e").groupby(lambda x: "ground" if (x < -0.4) else ("excited" if x < 0.0 else ("low-lying continuum" if (x < .06) else "ionized" ))).aggregate( pandas.np.sum )
    for i in g:
        g[i].loc["ionized"] += (1 - g[i].sum())
    return g

def ionization( indexedwf, cutoff = 0.0 ):
    r = indexedwf.copy()
    #r["wf"] = r["wf"].map( lambda x: abs(x)**2 )
    bf = lambda x:  "ionized" if x > cutoff else ("bound" if x > -.4 else "ground")
    g = r.set_index('e').groupby(bf)
    r = g.aggregate( pandas.np.sum )
    return {"ground": r["wf"].loc["ground"], "excited" : r["wf"].loc["bound"], "ionized" : r["wf"].loc["ionized"] + (1-r["wf"].sum()), "absorbed": 1- r["wf"].sum()}


def all_wfs_mapped( folder = "./" ):
    prototype = get_prototype(folder + "/prototype.csv")
    wfs = wavefunctions(folder)
    iwfs = [ indexed_wf(prototype, wfs[wf]) for wf in wfs]
    gnwfs= [ grouped_by("n", wf) for wf in iwfs]
    gewfs = [binned_group_by(.01, wf) for wf in iwfs]
    return (iwfs, gnwfs, gewfs)

def final_wf_mapped( folder = "./", bin_size = .01):
    prototype = get_prototype( folder + "/prototype.csv")
    iwf = indexed_wf( prototype, import_petsc_vec( folder +  "/wf_final.dat" ) )
    gnwf= grouped_by("n", iwf)
    gewf = binned_group_by(bin_size, iwf)
    return (iwf, gnwf, gewf)

def timeseries(fname, tfname = "time.dat", title="tsdata" ):
    with open(tfname, "rb") as f:
        time = numpy.fromfile(f, 'd', -1)
    with open(fname, "rb") as f:
        ef = numpy.fromfile(f, 'd', -1)
    return pandas.DataFrame( {title: ef}, index=time)

def efield( effname = "efield.dat" ):
    return timeseries( fname = effname ,title="efield")

def normalize( wf ):
    wfn = wf.copy()
    wfn = wfn / wfn.sum()
    return wfn

def photon_binned( prob, photon_energy, num_bins ):
    r = indexedwf.copy()
    minimum = 0.
    maximum = photon_energy * num_bins;
    diff = maximum - minimum
    #create the bins:
    bins = [ ((i + 1) * bin_size + minimum, (i + .5) * bin_size + minimum ) for i in range(int ( diff / bin_size ) + 1) ]
    bf = lambda x: binfunc( x, bins )
    g = r.set_index('e').groupby( bf )
    r = g.aggregate( pandas.np.sum )
    return pandas.DataFrame({"wf":r['wf']})
