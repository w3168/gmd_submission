Non-dimensional displacement fields at the surface are stored in paraview output format in
`surface_outputs.tar.gz` 
and csv format in `outputs_csv.tar.gz`
at 1 kyr intervals (0-10 kyr)

The peak negative radial (inward) displacement at each timestep can be found in
`displacement-sphere-burgers-3d-internalvariable-gmd-reflevel6.0-nz10perlayer-dt50.0years-bulk1.94-nondim.dat`

To run the spherical base case

`> python3 3d_sphere_burgers.py --reflevel 6 --DG0_layers 10 --dt_years 50 --bulk_shear_ratio 1.94 --viscosity_ratio 0.1 --lateral_visc --output_path /path/to/output/`


N.b. that we carried out this simulation in parallel on 832 Saphire Rapid CPUs on NCI's GADI HPC system.
