# Idealised 2-D viscoelastic loading problem in a square box
# =======================================================
#

from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
from gadopt.utility import vertical_component as vc
import argparse
import numpy as np
from mpi4py import MPI
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--ncells", default=360, type=float, help="Number of cells in the horizontal surface mesh", required=False)
parser.add_argument("--DG0_layers", default=20, type=int, help="Number of cells per layer for DG0 discretisation of background profiles", required=False)
parser.add_argument("--dt_years", default=50, type=float, help="Timestep in years", required=False)
parser.add_argument("--Tend", default=10e3, type=float, help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=1.94, type=float, help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--radial_visc", action='store_true', help="Use 1D viscosity profile")
parser.add_argument("--burgers", action='store_true', help="Use Burgers rheology")
parser.add_argument("--ramp_ice", action='store_true', help="Ramp ice up")
parser.add_argument("--write_output", action='store_true', help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str, help="Optional string to add to simulation name for outputs", required=False)
parser.add_argument("--output_path", default="/data/viscoelastic/internal_variable_adjoint/forward/", type=str, help="Optional output path", required=False)
args = parser.parse_args()

name = f"forward-cylinder-2d-internalvariable-dispvel-{args.optional_name}-1dvisc{args.radial_visc}-burgers{args.burgers}"

# +
# Set up geometry:
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
D = radius_values[0]-radius_values[-1]
radius_values_tilde = np.array(radius_values)/D
 
layer_height_list = []
DG0_layers = args.DG0_layers
nz_layers = [DG0_layers, DG0_layers, DG0_layers, DG0_layers]

for j in range(len(radius_values_tilde)-1):
    i = len(radius_values_tilde)-2 - j  # want to start at the bottom
    r = radius_values_tilde[i]
    h = r - radius_values_tilde[i+1]
    nz = nz_layers[i]
    dz = h / nz

    for i in range(nz):
        layer_height_list.append(dz)

# Construct a circle mesh and then extrude into a cylinder:
ncells = args.ncells
rmin = radius_values_tilde[-1]
surface_mesh = CircleManifoldMesh(ncells, radius=rmin, degree=2, name='surface_mesh')

mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_height_list),
    layer_height=layer_height_list,
    extrusion_type='radial'
)

mesh.cartesian = False
boundary = get_boundary_ids(mesh)
nz = f"{DG0_layers}perlayer"

ds = CombinedSurfaceMeasure(mesh, degree=6)

log("Area of annulus: ", assemble(Constant(1) * dx(domain=mesh)))
log("Length of top: ", assemble(Constant(1) * ds(boundary.top, domain=mesh)))
log("Length of bottom: ", assemble(Constant(1) * ds(boundary.bottom, domain=mesh)))

# -

# Set up function spaces:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
S = TensorFunctionSpace(mesh, "DQ", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG0 = FunctionSpace(mesh, "DG", 0)  # (Discontinuous) Stress tensor function space (tensor)
DG1 = FunctionSpace(mesh, "DG", 1)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the incremental displacement and pressure spaces to form
# a function space for the mixed Stokes problem, `Z`.

Z = MixedFunctionSpace([V, S])  # Mixed function space.

# We also specify functions to hold our solutions: `z` in the mixed
# function space, noting that a symbolic representation of the two
# parts – incremental displacement and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of `z` to *Incremental Displacement* and *Pressure*.
#
# We also need to initialise two functions `displacement` and `stress_old` that are used when timestepping the constitutive equation.

# +
z = Function(Z)  # A field over the mixed function space Z.
# Function to store the solutions:
u = Function(V)  # a field over the mixed function space Z.
m = Function(S, name="internal variable")
m_list = [m]
if args.burgers:
    m2 = Function(S, name="internal variable 2")
    m_list.append(m2)
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable  DOF:", S.dim())
log("Number of Velocity and internal variable DOF:", V.dim()+S.dim())

# Let's start initialising some parameters. First of all Firedrake has a helpful function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties.
# In this case the density, shear modulus and viscosity only vary in the vertical direction.
# We will approximate the series of layers using a smooth tanh function with a width of 20 km.
# The layer properties specified are from spada et al. (2011).
# N.b. that we have modified the viscosity of the Lithosphere viscosity from
# Spada et al. (2011) because we are using coarse grid resolution.


# +
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e40, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21

density_values_tilde = np.array(density_values)/density_scale
shear_modulus_values_tilde = np.array(shear_modulus_values)/shear_modulus_scale
viscosity_values_tilde = np.array(viscosity_values)/viscosity_scale


def initialise_background_field(field, background_values):
    for i in range(0, len(background_values)):
        field.interpolate(conditional(vc(X) >= radius_values_tilde[i+1],
                          conditional(vc(X) <= radius_values_tilde[i],
                          background_values[i], field), field))



density = Function(DG0, name="density")
initialise_background_field(density, density_values_tilde)

shear_modulus = Function(DG0, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values_tilde)

# if Pseudo incompressible set bulk modulus to a constant...
# Otherwise use same jumps from shear modulus multiplied by a factor

bulk_modulus = Function(DG0, name="bulk modulus")
initialise_background_field(bulk_modulus, shear_modulus_values_tilde)

background_viscosity = Function(DG1, name="background viscosity")
initialise_background_field(background_viscosity, viscosity_values_tilde)

# Defined lateral viscosity regions
def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(viscosity):
    heterogenous_viscosity_field = Function(viscosity.function_space(), name='viscosity')
    antarctica_x, antarctica_y = -2e6/D, -5.5e6/D

    low_visc = 1e20/viscosity_scale
    high_visc = 1e22/viscosity_scale

    low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6/D, 0.5e6/D, -0.4)
    heterogenous_viscosity_field.interpolate(low_visc*low_viscosity_antarctica + viscosity * (1-low_viscosity_antarctica))

    llsvp1_x, llsvp1_y = 3.5e6/D, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6/D, 1e6/D, 0)
    heterogenous_viscosity_field.interpolate(low_visc*llsvp1 + heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6/D, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6/D, 1e6/D, 0)
    heterogenous_viscosity_field.interpolate(low_visc*llsvp2 + heterogenous_viscosity_field * (1-llsvp2))

    slab_x, slab_y = 3e6/D, 4.5e6/D
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6/D, 0.35e6/D, 0.7)
    heterogenous_viscosity_field.interpolate(high_visc*slab + heterogenous_viscosity_field * (1-slab))

    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6/D
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6/D, 0.5e6/D, 0.2)
    heterogenous_viscosity_field.interpolate(high_visc*high_viscosity_craton + heterogenous_viscosity_field * (1-high_viscosity_craton))
    
    heterogenous_viscosity_field.interpolate(conditional(vc(X)>radius_values_tilde[1], viscosity, heterogenous_viscosity_field))

    return heterogenous_viscosity_field

if args.radial_visc:
    viscosity = Function(background_viscosity, name='viscosity').assign(background_viscosity)
else:
    viscosity = setup_heterogenous_viscosity(background_viscosity)

# -

# Next let's define the length of our time step. If we want to accurately resolve the elastic response we should choose a
# timestep lower than the Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation
# to 'catch up' with the initial, instantaneous elastic deformation.
#
# Let's print out the Maxwell time for each layer

year_in_seconds = 8.64e4 * 365.25
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale
for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(layer_visc/layer_mu/year_in_seconds):.0f} years")
    log(f"Ratio to characteristic maxwell time: {float(layer_visc/layer_mu/characteristic_maxwell_time)}")


# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds/ characteristic_maxwell_time)

dt_years = args.dt_years
dt = Constant(dt_years * year_in_seconds/characteristic_maxwell_time)
Tend_years = args.Tend
Tend = Constant(Tend_years * year_in_seconds/characteristic_maxwell_time)
dt_out_years = dt_years
dt_out = Constant(dt_out_years * year_in_seconds/characteristic_maxwell_time)

max_timesteps = round((Tend - Tstart * year_in_seconds/characteristic_maxwell_time) / dt)
log("max timesteps: ", max_timesteps)

output_frequency = round(dt_out / dt)
log("output_frequency:", output_frequency)
log(f"dt: {float(dt)} maxwell times")
log(f"dt: {float(dt * characteristic_maxwell_time / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} maxwell times")
log(f"Simulation end time: {Tend} maxwell times")
log(f"Simulation end time: {float(Tend * characteristic_maxwell_time / year_in_seconds)} years")
# -


# Initialise ice loading
rho_ice = 931 / density_scale
g = 9.815
B_mu = Constant(density_scale * D * g / shear_modulus_scale)
log("Ratio of buoyancy/shear = rho g D / mu = ", float(B_mu))
Hice1 = 1000 / D
Hice2 = 2000 / D
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
surface_dx_smooth = 200*1e3
ncells_smooth = 2*pi*radius_values[0] / surface_dx_smooth
surface_resolution_radians_smooth = 2*pi / ncells_smooth
colatitude = atan2(X[0], X[1])
disc1_centre = (2*pi/360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians_smooth)))
disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians_smooth)))

P1 = FunctionSpace(mesh, "CG", 1)
discfunc = Function(P1).interpolate(D*(Hice1*disc1+Hice2*disc2))
discfile = VTKFile(f"{args.output_path}discfile.pvd").write(discfunc)

if args.ramp_ice:
    t1_load = 90e3 * year_in_seconds / characteristic_maxwell_time
    t2_load = 100e3 * year_in_seconds / characteristic_maxwell_time
    ramp_after_t1 = conditional(
        time < t2_load, 1 - (time - t1_load) / (t2_load - t1_load), 0
    )
    ramp = conditional(time < t1_load, time / t1_load, ramp_after_t1)
else:
    ramp = Constant(1)
ice_load = ramp * B_mu * rho_ice * (Hice1 * disc1 + Hice2 * disc2)

# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and
# side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing
# the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as
# indicating this is a free surface.
#
# +
# Setup boundary conditions
stokes_bcs = {
    boundary.bottom: {'un': 0},
    boundary.top: {'normal_stress': ice_load, 'free_surface': {}},
}

gd = GeodynamicalDiagnostics(z, density, boundary.bottom, boundary.top)
# -

if args.burgers:
    shearmod_list = [0.5*shear_modulus, 0.5*shear_modulus]
    visc_list = [0.5*viscosity, 0.1*0.5*viscosity]
else:
    shearmod_list = [shear_modulus]
    visc_list = [viscosity] 
# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.

#approximation = QuasiCompressibleInternalVariableApproximation(bulk_modulus=bulk_modulus, density=density, shear_modulus=shearmod_list, viscosity=visc_list, B_mu=B_mu, bulk_shear_ratio=args.bulk_shear_ratio)
approximation = CompressibleInternalVariableApproximation(bulk_modulus=bulk_modulus, density=density, shear_modulus=shearmod_list, viscosity=visc_list, B_mu=B_mu, bulk_shear_ratio=args.bulk_shear_ratio)

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields `z` and various fields
# needed for the solve along with the approximation, timestep and boundary conditions.
#

direct_stokes_solver_parameters = {
    "snes_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

iterative_parameters = {"mat_type": "matfree",
                            "snes_type": "ksponly",
                            "ksp_type": "gmres",
                            "ksp_rtol": 1e-5,
                            "ksp_converged_reason": None,
    #                        "ksp_monitor": None,
                            "pc_type": "python",
                            "pc_python_type": "firedrake.AssembledPC",
                            "assembled_pc_type": "gamg",
                            "assembled_mg_levels_pc_type": "sor",
                            "assembled_pc_gamg_threshold": 0.01,
                            "assembled_pc_gamg_square_graph": 100,
                            "assembled_pc_gamg_coarse_eq_limit": 1000,
                            "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
                            }


V_nullspace = rigid_body_modes(V, rotational=True)
V_near_nullspace = rigid_body_modes(V, rotational=True, translations=[0, 1])

coupled_solver = InternalVariableSolver(u, approximation, dt=dt, internal_variables=m_list, bcs=stokes_bcs,
#                                       solver_parameters=direct_stokes_solver_parameters,
                                        solver_parameters=iterative_parameters,
                                       nullspace=V_nullspace, transpose_nullspace=V_nullspace,
                                       near_nullspace=V_near_nullspace)


# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create output file
OUTPUT = args.write_output
vertical_displacement = Function(V.sub(1), name="radial displacement")  # Function to store vertical displacement for output
disp_x = Function(V.sub(0), name="displacement x")  # Function to store x displacement for output
disp_y = Function(V.sub(1), name="displacement y")  # Function to store y displacement for output
f = Function(V).interpolate(as_vector([X[0], X[1]]))
bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, boundary.top)

surface_x = f.sub(0).dat.data_ro_with_halos[bc_displacement.nodes]

surface_x_all = f.sub(0).comm.gather(surface_x)
surface_y = f.sub(1).dat.data_ro_with_halos[bc_displacement.nodes]
surface_y_all = f.sub(1).comm.gather(surface_y)
displacement_df = pd.DataFrame()

if MPI.COMM_WORLD.rank == 0:
    surface_x_concat = np.concatenate(surface_x_all)
    displacement_df['surface_x'] = surface_x_concat
    surface_y_concat = np.concatenate(surface_y_all)
    displacement_df['surface_y'] = surface_y_concat

velocity = Function(u, name="velocity")
disp_old = Function(u, name="old_disp").assign(u)

if OUTPUT:
    log("hello visco output")
    visc_file = VTKFile(f"{args.output_path}{name}-visc.pvd")
    visc_file.write(viscosity)
    output_file = VTKFile(f"{args.output_path}{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.pvd")
    output_file.write(u, *m_list, vertical_displacement, velocity)

plog = ParameterLog(args.output_path+"params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)


checkpoint_filename = f"{args.output_path}{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulktoshear{args.bulk_shear_ratio}-nondim-chk.h5"

displacement_filename = f"{args.output_path}min-displacement-{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.dat"
surface_displacement_filename = f"{args.output_path}surface-displacement-{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.dat"


# Initial displacement at time zero is zero
displacement_min_array = [[0.0, 0.0]]

objective_filename = f"{args.output_path}displacement-objective-{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.h5"
objective_checkpoint_file = CheckpointFile(objective_filename, "w")
objective_checkpoint_file.save_mesh(mesh)
# -

# Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter.
# At each step we call `solve` to calculate the incremental displacement and pressure fields. This
# will update the displacement at the surface and stress values accounting for the time dependent
# Maxwell consitutive equation.
forward_stage = PETSc.Log.Stage("forward")

for timestep in range(1, max_timesteps+1):
    # update time first so that ice load begins
    time.assign(time+dt)
    with forward_stage:
        coupled_solver.solve()
    velocity.interpolate((u - disp_old)/dt)
    objective_checkpoint_file.save_function(u, name="Displacement", idx=timestep)
    objective_checkpoint_file.save_function(velocity, name="Velocity", idx=timestep)
    disp_old.assign(u)

    # Log diagnostics:
    # Compute diagnostics:
    # output dimensional vertical displacement
    vertical_displacement.interpolate(vc(u)*D)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (-ve) displacement", displacement_min)
    displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
    displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (+ve) displacement", displacement_max)
    displacement_min_array.append([float(characteristic_maxwell_time*time.dat.data[0]/year_in_seconds), displacement_min])

    disp_x.interpolate(u[0]*D)
    surface_disp_x = disp_x.dat.data_ro_with_halos[bc_displacement.nodes]
    surface_disp_x_all = disp_x.comm.gather(surface_disp_x)
    disp_y.interpolate(u[1]*D)
    surface_disp_y = disp_y.dat.data_ro_with_halos[bc_displacement.nodes]
    surface_disp_y_all = disp_y.comm.gather(surface_disp_y)

    if MPI.COMM_WORLD.rank == 0:
        surface_disp_x_concat = np.concatenate(surface_disp_x_all)
        displacement_df[f'surface_disp_x_step{timestep}'] = surface_disp_x_concat
        
        surface_disp_y_concat = np.concatenate(surface_disp_y_all)
        displacement_df[f'surface_disp_y_step{timestep}'] = surface_disp_y_concat

#    disp_norm_L2surf = assemble((z.subfunctions[0][vertical_component])**2 * ds(boundary.top))
 #   log("L2 surface norm displacement", disp_norm_L2surf)

  #  disp_norm_L1surf = assemble(abs(z.subfunctions[0][vertical_component]) * ds(boundary.top))
   # log("L1 surface norm displacement", disp_norm_L1surf)

    #integrated_disp = assemble(z.subfunctions[0][vertical_component] * ds(boundary.top))
    #log("Integrated displacement", integrated_disp)

    if timestep % output_frequency == 0:
        log("timestep", timestep)
        displacement_df.to_csv(surface_displacement_filename)

        if OUTPUT:
            output_file.write(u, *m_list, vertical_displacement, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Stokes")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

        plog.log_str(f"{timestep} {float(time)} {float(dt)} "
                     f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} "
                     )

objective_checkpoint_file.close()
# Let's use the python package *PyVista* to plot the magnitude of the displacement field through time.
# We will use the calculated displacement to artifically scale the mesh. We have exaggerated the stretching
# by a factor of 1500, **BUT...** it is important to remember this is just for ease of visualisation -
# the mesh is not moving in reality!

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import pyvista as pv
#
# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
#
# # Open a gif
# plotter.open_gif("displacement_warp.gif")
#
# # Make a colour map
# boring_cmap = plt.get_cmap("viridis", 25)
#
# for i in range(len(reader.time_values)):
#     reader.set_active_time_point(i)
#     data = reader.read()[0]
#
#     # Artificially warp the output data in the vertical direction by the free surface height
#     # Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=1500)
#     arrows = data.glyph(orient="Incremental Displacement", scale="Incremental Displacement", factor=400000, tolerance=0.05)
#     plotter.add_mesh(arrows, color="white", lighting=False)
#
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="displacement",
#         component=None,
#         lighting=False,
#         show_edges=False,
#         clim=[0, 70],
#         cmap=boring_cmap,
#         scalar_bar_args={
#             "title": 'Displacement (m)',
#             "position_x": 0.8,
#             "position_y": 0.2,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#     )
#
#     # Fix camera in default position otherwise mesh appears to jump around!
#     plotter.camera_position = [(750000.0, 1445500.0, 6291991.008627122),
#                         (750000.0, 1445500.0, 0.0),
#                         (0.0, 1.0, 0.0)]
#     plotter.add_text(f"Time: {i*2000:6} years", name='time-label')
#     plotter.write_frame()
#
#     if i == len(reader.time_values)-1:
#         # Write end frame multiple times to give a pause before gif starts again!
#         for j in range(20):
#             plotter.write_frame()
#
#     plotter.clear()
#
# # Closes and finalizes movie
# plotter.close()
# -
# Looking at the animation, we can see that as the weight of the ice load builds up the mantle deforms,
# pushing up material away from the ice load. If we kept the ice load fixed this forebulge will
# eventually grow enough that it balances the weight of the ice, i.e the mantle is in isostatic
# equilbrium and the deformation due to the ice load stops. At 100 thousand years when the ice is removed
# the topographic highs associated with forebulges are now out of equilibrium so the flow of material
# in the mantle reverses back towards the previously glaciated region.

# ![SegmentLocal](displacement_warp.gif "segment")

# References
# ----------
# Cathles L.M. (1975). *Viscosity of the Earth's Mantle*, Princeton University Press.
#
# Dahlen F. A. and Tromp J. (1998). *Theoretical Global Seismology*, Princeton University Press.
#
# Ranalli, G. (1995). Rheology of the Earth. Springer Science & Business Media.
#
# Weerdesteijn, M. F., Naliboff, J. B., Conrad, C. P., Reusen, J. M., Steffen, R., Heister, T., &
# Zhang, J. (2023). *Modeling viscoelastic solid earth deformation due to ice age and contemporary
# glacial mass changes in ASPECT*. Geochemistry, Geophysics, Geosystems.
#
# Wu P., Peltier W. R. (1982). *Viscous gravitational relaxation*, Geophysical Journal International.
#
# Zhong, S., Paulson, A., & Wahr, J. (2003). Three-dimensional finite-element modelling of Earth’s
# viscoelastic deformation: effects of lateral variations in lithospheric thickness. Geophysical
# Journal International.
