#!/bin/bash
#PBS -N internalvariable
#PBS -P vo05
#PBS -q normalsr
#PBS -l walltime=48:00:00
#PBS -l mem=4000GB
#PBS -l ncpus=832
#PBS -l jobfs=100GB
#PBS -l storage=scratch/xd2+gdata/xd2+scratch/vo05+gdata/vo05+gdata/fp50
#PBS -l wd

export MY_GADOPT=$HOME/g-adopt
export PYTHONPATH="${PYTHONPATH}:/g/data/xd2/ws9229/Irksome"
module use /g/data/fp50/modules
module load firedrake

export MPLCONFIGDIR=${PBS_JOBFS}/MPL_DIR
mpiexec --map-by ppr:1:node -np $PBS_NNODES  python3 -c "import matplotlib.pyplot as plt"

export GADOPT_LOGLEVEL="DEBUG"
cd /g/data/xd2/ws9229/gmd_submission/3.2-3.3-cartesian-benchmarks/
dglayers=10
dx=5
bulk_shear_ratio=1000
dt=10
Tend=200
dt_out=50
output_path="/g/data/xd2/ws9229/gmd_submission/3.2-3.3-cartesian-benchmarks/3.2-incompressible-maxwell/short-1d/"



# Linear maxwell incompressible short
mpiexec -np ${PBS_NCPUS} python3 3d_weerdesteijn_coupled.py --short_simulation --dx $dx --refined_surface --DG0_layers $dglayers --dt_years $dt --bulk_shear_ratio $bulk_shear_ratio --Tend $Tend --dt_out_years $dt_out --output_path $output_path --optional_name "gmd" &> 11.03.26_weerdesteijn_internalvariable_110ka_dt${dt}yr_dtout10ka_normalsr${PBS_NCPUS}cores_mem4000gb_dx${dx}to200km_refinedsurface_DG0nzperlayer${dglayers}_icetanh1km_bulktoshear${bulk_shear_ratio}_coupled_1d_short

