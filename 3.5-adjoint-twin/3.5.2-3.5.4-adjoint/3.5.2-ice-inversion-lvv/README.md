This folder contains files associated with the adjoint twin experiment inverting for the initial
ice load using the correct laterally varying viscosity structure.

To run this case use the python script in the parent directory

`python adjoint.py --controls ice --true_visc`

Values of the objective function (functional) at each optimisation iteration are provided in
`adjoint-cylinder-2d-internalvariable-ctypeice-gmd_ice_inversion_lvv_functional.txt`

Updated ice thickness at each iteration of the optimisatiion can be viewed in paraview formatting
by extracting the files in `updated_ice_paraview.tar.gz` and using
`adjoint-cylinder-2d-internalvariable-ctypeice-gmd_ice_inversion_lvv_surface_ice.pvd` with Paraview.

CSV version of the same updated ice thickness results are provided in `updated_ice.tar.gz`
