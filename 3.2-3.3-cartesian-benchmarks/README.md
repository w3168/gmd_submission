Two G-ADOPT runscripts are provided to reproduce results from Section 3.2 and 3.3. 

The incompressible cases from Section 3.2 and the power-law rheology from Section 3.3
use the coupled formulation (`3d_weerdesteijn_coupled.py`). All other cases use the
substitution approach (`3d_weerdesteijn.py`)

Both script takes a set of input arguments to specify timestep, resolution etc. Specific
argument values for each case are documented in the respective subfolders and also in
the example batch job script files used to run the simulations on NCI's GADI HPC system.

Surface displacement outputs can be found in the respective subfolders for each case.
