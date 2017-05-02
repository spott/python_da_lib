# python_da_lib

Data analysis library for reading data from EBSS.  Includes "perturbation.py" which is a adaptive integrator for time dependent perturbation theory.

## Requirements:
 * A fairly modern python 3 version.  Most of the lib requires 3.6 (f strings), however perturbation.py should run on 3.5 or 3.4.
 * petsc4py with petsc and complex value support.
 * mpi4py
 * core
 * numpy
 * pandas
 
## Usage:

  `--hamiltonian_folder` - the folder that the hamiltonian (dipole matrix and prototype -- which is the map from (n,l,j,m,e) to index number in the basis)
  `--efield_folder` The folder that has the ab-initio run that includes the efield values.  This *should* be removed and done in file eventually.
  `--steps` size of each adaptive "big" step. Too small, and unnecessary work is done (the tree can only be so shallow, too big and unnecessary work is done.  Should be some 3^n... though not strictly necessary 
  `--save_steps` when to save (in number of "dt" steps).  Currently at 1, and I recommend it stays that way.  Will only save when the adaptive step is done anyways.
  `--out_file` the hdf file to save to
  `--key` the key in the hdf file.
  `--relative_error` the maximum error that average(|error|) / max(|error|) can be.  set to 1e-3 by default.
  `--absolute_error` the maximum error that average(|error|) can be.  Currently error checking is either/or, to prevent spending too much time on early results
  
