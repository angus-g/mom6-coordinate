#!/bin/bash

# patch MOM_remapping.F90 to expose everything as public (otherwise
# the interface can't access anything)
sed -e 's/private//g' mom6/src/ALE/MOM_remapping.F90 > MOM_remapping.F90

# build the original fortran source to modules
# this is a pretty ugly way of doing it, but it works
# we've hardcoded the source files in an order that satisfies dependencies
# ideally we'd use a proper build system (ninja, make)
# but hopefully we don't touch the actual remapping core
files="MOM_error_handler.F90 mom6/src/framework/MOM_string_functions.F90 mom6/src/ALE/regrid_solvers.F90 mom6/src/ALE/polynomial_functions.F90 mom6/src/ALE/regrid_edge_slopes.F90 mom6/src/ALE/regrid_edge_values.F90 mom6/src/ALE/PCM_functions.F90 mom6/src/ALE/PLM_functions.F90 mom6/src/ALE/PPM_functions.F90 mom6/src/ALE/PQM_functions.F90 MOM_remapping.F90"
fflags="-fcray-pointer -fdefault-double-8 -fdefault-real-8 -Waliasing -ffree-line-length-none -fno-range-check -O3 -fPIC"
gfortran ${fflags} -I. -c ${files}

# if the API has changed, regenerate (note that changes are required!)
#f90wrap -m remapping --only remapping_core_h -- mom6/src/ALE/MOM_remapping.F90

# build the interface
f2py --f90flags="${fflags}" -c -m _remapping f90wrap_*.f90 *.o
