Short simulation, compressible, Burgers rheology (viscosity 2 / viscosity 1 = 0.1)

Non-dimensional displacement fields at the surface are stored in paraview output format in
`surface_outputs.tar.gz` and csv format in `surface_outputs_csv.tar.gz`
at 50 yr intervals (0-200 yr). To redimensionalise the displacement fields multiply by
the depth of the domain, 2891 km.

The peak negative radial (inward) displacement in metres at each timestep can be found in
`displacement-weerdesteijn-3d-iv-burgersTrue-gmd-refinedsurfaceTrue-dx5.0km-nz10perlayer-dt10.0years-bulk1.94-nondim.dat`

To run the case use the `3d_weerdesteijn.py` script from the parent directory
`gmd_submission/3.2-3.3-cartesian-benchmarks`

`> python3 3d_weerdesteijn.py --burgers --viscosity_ratio 0.1 --short_simulation --refined_surface --dx 5 --DG0_layers 10 --dt_years 10 --bulk_shear_ratio 1.94 --Tend 200 --dt_out_years 50 --output_path /path/to/output/`

N.b. that we carried out this simulation in parallel on 832 Saphire Rapid CPUs on NCI's
GADI HPC system. An example job script is provided in this folder.
