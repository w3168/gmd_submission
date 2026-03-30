This folder contains files associated with the adjoint twin experiment inverting for the
viscosity structure assuming the correct ice load.

To run this case use the python script in the parent directory

`python adjoint.py --controls viscosity --true_ice`

Values of the objective function (functional) at each optimisation iteration are provided in
`adjoint-cylinder-2d-internalvariable-ctypeviscosity-gmd_viscosity_inversion_functional.txt`

Updated viscosity fields at iterations 0, 1, 5, 10, 50 and 100 of the optimisatiion are provided in
`updated_viscosity.tar.gz`
