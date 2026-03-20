Long simulation, incompressible, Maxwell rheology, 1D viscosity profile

Non-dimensional displacement fields at the surface are stored in paraview output format in
`surface_outputs.tar.gz` and csv format in `surface_outputs_csv.tar.gz`
at 10 kyr intervals (0-110 kyr). To redimensionalise the displacement fields multiply by
the depth of the domain, 2891 km.

The peak negative radial (inward) displacement in metres at each timestep can be found in
`displacement-weerdesteijn-3d-iv-burgersFalse-gmd-refinedsurfaceTrue-dx5.0km-nz10perlayer-dt1000.0years-bulk1000-powerFalse-nondim.dat`

To run the case use the `3d_weerdesteijn_coupled.py` script from the parent directory
`gmd_submission/3.2-3.3-cartesian-benchmarks`

`> python3 3d_weerdesteijn_coupled.py --refined_surface --dx 5 --DG0_layers 10 --dt_years 1000 --bulk_shear_ratio 1000 --Tend 110e3 --dt_out_years 10e3 --output_path /path/to/output/`

N.b. that we carried out this simulation in parallel on 832 Saphire Rapid CPUs on NCI's
GADI HPC system. An example job script is provided in this folder.
