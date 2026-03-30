This folder contains files associated with the adjoint twin experiment inverting for
both the viscosity structure and initial ice load.

To run this case use the python script in the parent directory

`python adjoint.py --controls both`

Values of the objective function (functional) at each optimisation iteration are provided in
`adjoint-cylinder-2d-internalvariable-ctypeboth-gmd_both_inversion_functional.txt`

Updated viscosity fields at iterations 0, 1, 5, 10, 50 and 100 of the optimisation are provided in
`updated_viscosity.tar.gz`

Updated ice thickness at each iteration of the optimisatiion can be viewed in paraview formatting
by extracting the files in `updated_ice_paraview.tar.gz` and using
`adjoint-cylinder-2d-internalvariable-ctypeboth-gmd_both_inversion_surface_ice.pvd` with Paraview.

CSV version of the same updated ice thickness results are provided in `updated_ice.tar.gz`
