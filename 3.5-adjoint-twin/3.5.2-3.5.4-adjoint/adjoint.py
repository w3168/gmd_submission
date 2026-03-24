"""
This runs the optimisation portion of the adjoint test case. A forward run first sets up
the tape with the adjoint information, then a misfit functional is constructed to be
used as the goal condition for nonlinear optimisation using ROL.

annulus_taylor_test is also added to this script for testing the correctness of the gradient for the inverse problem.
    taylor_test(alpha_T, alpha_u, alpha_d, alpha_s):
            alpha_T (float): The coefficient of the temperature misfit term.
            alpha_u (float): The coefficient of the velocity misfit term.
            alpha_d (float): The coefficient of the initial condition damping term.
            alpha_s (float): The coefficient of the smoothing term.
            float: The minimum convergence rate from the Taylor test. (Should be close to 2)
"""
from gadopt import *
from gadopt.inverse import *
import numpy as np
import pandas as pd
# from checkpoint_schedules import SingleDiskStorageSchedule
import sys
from mpi4py import MPI
import argparse
from gadopt.utility import CombinedSurfaceMeasure
from gadopt.utility import vertical_component as vc
from gadopt.utility import upward_normal
parser = argparse.ArgumentParser()
parser.add_argument("--ncells", default=360, type=float, help="Number of cells in the horizontal surface mesh", required=False)
parser.add_argument("--DG0_layers", default=20, type=int, help="Number of cells per layer for DG0 discretisation of background profiles", required=False)
parser.add_argument("--dt_years", default=50, type=float, help="Timestep in years", required=False)
parser.add_argument("--Tend", default=10e3, type=float, help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=1.94, type=float, help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--write_output", action='store_true', help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str, help="Optional string to add to simulation name for outputs", required=False)
parser.add_argument("--output_path", default="/data/viscoelastic/internal_variable_adjoint/adjoint/", type=str, help="Optional output path", required=False)
parser.add_argument("--controls", default="ice", type=str, help="Specify which control, ice/viscosity/both", required=False)
parser.add_argument("--ice_checkpoint", default=None, type=str, help="Ice checkpoint", required=False)
parser.add_argument("--viscosity_checkpoint", default=None, type=str, help="Viscosity checkpoint", required=False)
parser.add_argument("--ice_smoothing", default=0.0, type=float, help="ice smoothing factor", required=False)
parser.add_argument("--ice_damping", default=0.0, type=float, help="ice_damping factor", required=False)
parser.add_argument("--visc_smoothing", default=0.0, type=float, help="viscosity smoothing factor", required=False)
parser.add_argument("--visc_damping", default=0.0, type=float, help="viscosity damping factor", required=False)
parser.add_argument("--true_ice", action='store_true', help="use actual ice")
parser.add_argument("--true_visc", action='store_true', help="use actual viscosity")
parser.add_argument("--opt_its", default=100, type=int, help="Number of optimisation iterations", required=False)
parser.add_argument("--opt_max_rad", default=1e20, type=float, help="Maximum radius of linmore algorithm", required=False)
parser.add_argument("--burgers", action='store_true', help="Burgers model")
parser.add_argument("--burg_ratio", default=0.1, type=float, help="eta2 / eta1 burgers ratio", required=False)
parser.add_argument("--time_integrated_sensitivity", action='store_true', help="Set objective to a time integrated displacement and velocity")
parser.add_argument("--final_sensitivity", action='store_true', help="Set objective to final displacement and velocity")
parser.add_argument("--obj_radial", action='store_true', help="Only use radial component in objective function")
parser.add_argument("--obj_tangential", action='store_true', help="Only use tangential component in objective function")
args = parser.parse_args()

name = f"adjoint-cylinder-2d-internalvariable-ctype{args.controls}-{args.optional_name}"



def inverse(): #alpha_T=1e0, alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1):

    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem() #alpha_u=alpha_u, alpha_d=alpha_d, alpha_s=alpha_s)

    
    minimisation_problem = MinimizationProblem(inverse_problem["reduced_functional"], bounds=inverse_problem["bounds"])

    minimisation_parameters["Status Test"]["Iteration Limit"] = args.opt_its
    minimisation_parameters["Step"]["Trust Region"]["Maximum Radius"] = args.opt_max_rad

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
    )

    
    # Restart file for optimisation...
    functional_values = []

    optimiser.run()
    

    # If we're performing multiple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()

def replay_tape(): #alpha_T, alpha_u, alpha_d, alpha_s):
    """
    Perform a Taylor test to verify the correctness of the gradient for the inverse problem.

    This function calls a main function to populate the tape for the inverse problem
    with specified regularization parameters, generates a random perturbation for the control variable,
    and performs a Taylor test to ensure the gradient is correct. Finally, it ensures that annotations
    are switched back on for any subsequent tests.

    Returns:
        minconv (float): The minimum convergence rate from the Taylor test.
    """



    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem() #alpha_T, alpha_u, alpha_d, alpha_s)
    

    Jval = inverse_problem["reduced_functional"](inverse_problem["control"])
    log(Jval)
    # If we're performing mulitple successive tests we want
    # to ensure the annotations are switched back on for the next code to use them
    continue_annotation()

    return Jval

def check_taylor_test(): #alpha_T, alpha_u, alpha_d, alpha_s):
    """
    Perform a Taylor test to verify the correctness of the gradient for the inverse problem.

    This function calls a main function to populate the tape for the inverse problem
    with specified regularization parameters, generates a random perturbation for the control variable,
    and performs a Taylor test to ensure the gradient is correct. Finally, it ensures that annotations
    are switched back on for any subsequent tests.

    Returns:
        minconv (float): The minimum convergence rate from the Taylor test.
    """



    # For solving the inverse problem we the reduced functional, any callback functions,
    # and the initial guess for the control variable
    inverse_problem = generate_inverse_problem() #alpha_T, alpha_u, alpha_d, alpha_s)
    
    # generate perturbation for the control variable
    h = Function(inverse_problem["control"][0].function_space(), name="perturbation")
    h.dat.data[:] = np.random.random(h.dat.data.shape)

    # Perform a taylor test to ensure the gradient is correct
    minconv = taylor_test(
        inverse_problem["reduced_functional"],
        inverse_problem["control"],
        h
    )

    # If we're performing mulitple successive tests we want
    # to ensure the annotations are switched back on for the next code to use them
    continue_annotation()

    return minconv

def check_speed(): 
    """
    Time forward and derivative calculation using petsc stages.

    Need to set in terminal
    >>> export PETSC_OPTIONS="-log_view"

    """

    forward_stage = PETSc.Log.Stage("forward")
    adjoint_stage = PETSc.Log.Stage("adjoint")
    inverse_problem = generate_inverse_problem() #alpha_T, alpha_u, alpha_d, alpha_s)    
    
    # Time second forward and derivative in case some caching perfomed...
    with forward_stage:
        forward = inverse_problem["reduced_functional"](inverse_problem["control"][0])
    with adjoint_stage:
        deriv = inverse_problem["reduced_functional"].derivative()

    return deriv 

def check_derivative(): 
    """
    plot out derivative of ice and viscosity
    """

    inverse_problem = generate_inverse_problem()
    
    deriv = inverse_problem["reduced_functional"].derivative(apply_riesz=True)
    log(deriv)

    deriv_file = VTKFile(f"{args.output_path}/{name}-derivative.pvd").write(*deriv)

    return deriv 


def generate_inverse_problem(): # alpha_T=1.0, alpha_u=-1, alpha_d=-1, alpha_s=-1):
    """
    Use adjoint-based optimisation to solve for the initial condition of the cylindrical
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Get working tape
    tape = get_working_tape()
    tape.clear_tape()

    # # Writing to disk for block variables
    # enable_disk_checkpointing()

    # # Using SingleDiskStorageSchedule
    # if any([alpha_T > 0, alpha_u > 0]):
    #     tape.enable_checkpointing(SingleDiskStorageSchedule())

    # If we are not annotating, let's switch on taping
    if not annotate_tape():
        continue_annotation()
    # +
    # Set up geometry:
    radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
    D = radius_values[0]-radius_values[-1]
    radius_values_tilde = np.array(radius_values)/D
     
    DG0_layers = args.DG0_layers

    checkpoint_file = "../3.5.1-forward/outputs/displacement-objective-forward-cylinder-2d-internalvariable-dispvel-gmd-1dviscFalse-burgersFalse-ncells360-nz20perlayer-dt50years-bulk1.94-nondim.h5"

    with CheckpointFile(checkpoint_file, 'r') as afile:
        mesh = afile.load_mesh(name='surface_mesh_extruded')
        ncells=360
        # surface ice mesh needs to be first order for interpolation to work
        surface_mesh = CircleManifoldMesh(ncells, radius=radius_values_tilde[0], degree=1, name='surface_mesh')
        # Load surface mesh for ice control

    mesh.cartesian = False
    k_vec = upward_normal(mesh)
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
    P1 = FunctionSpace(mesh, "CG", 1)  
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)  
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
    # Function to store the solutions:
    u = Function(V)  # a field over the mixed function space Z.
    u_dim = Function(u, name="u dimensional")
    m = Function(S, name="internal variable")
    m_list = [m]
    if args.burgers:
        m2 = Function(S, name="internal variable 2")
        m_list.append(m2)
        burg_ratio = Function(R).assign(args.burg_ratio)
        if args.controls == "viscosity" or args.controls ==  "both":
            control3 = Control(burg_ratio)
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
    viscosity_values = [1e25, 1e21, 1e21, 2e21]

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

    background_viscosity = Function(DG0, name="background viscosity")
    initialise_background_field(background_viscosity, viscosity_values_tilde)

    background_viscosity_DG1 = Function(DG1, name="background viscosity DG1").interpolate(background_viscosity)

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
        
        # reset lithospheric viscosity
        heterogenous_viscosity_field.interpolate(conditional(vc(X)>radius_values_tilde[1], viscosity, heterogenous_viscosity_field))

        return heterogenous_viscosity_field


    target_viscosity = setup_heterogenous_viscosity(background_viscosity_DG1)

    # defining the control
    if args.viscosity_checkpoint:
        with CheckpointFile(args.viscosity_checkpoint, 'r') as afile:
            control_viscosity = afile.load_function(mesh, name="control viscosity")
            original_viscosity = afile.load_function(mesh, name="control viscosity")
    else:
        control_viscosity = Function(P1, name="control viscosity")
        original_viscosity = Function(P1, name="control viscosity")

    if args.controls == "viscosity" or args.controls ==  "both":
        control1 = Control(control_viscosity)
#        adj_visc_file = File(f"{args.output_path}{name}_adjvisc.pvd")
#        tape.add_block(DiagnosticBlock(adj_visc_file, control_viscosity))

    if args.true_visc:
        viscosity = target_viscosity
    else:
        viscosity = background_viscosity * 10**control_viscosity

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
    dt_out_years = 1e3
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

    #P1 = FunctionSpace(mesh, "CG", 1)

    #discfunc = Function(P1).interpolate(disc)
    
    #discfile = VTKFile(f"{args.output_path}discfile.pvd").write(discfunc)
    target_normalised_ice_thickness = Function(P1, name="target normalised ice thickness")
    target_normalised_ice_thickness.interpolate(disc1 + Hice2/Hice1 * disc2)
    
    # defining the control
    if args.ice_checkpoint:
        print("hello ice checkpoint")
        with CheckpointFile(args.ice_checkpoint, 'r') as afile:
            # Might not work? if wrong mesh...
            control_ice_thickness = afile.load_function(mesh, name="control normalised ice thickness")
            original_ice_thickness = afile.load_function(mesh, name="control normalised ice thickness")
    else:
        P1_surf = FunctionSpace(surface_mesh, "CG", 1)
        control_ice_thickness_surf = Function(P1_surf, name="control normalised ice thickness surf")
        control_ice_thickness = Function(P1, name="control normalised ice thickness")
        original_ice_thickness = Function(P1, name="control normalised ice thickness")

    # the ice thickness that will be actually used in simulation
    if args.true_ice:
        control_ice_thickness_surf.interpolate(target_normalised_ice_thickness, allow_missing_dofs=True) 
    
    if args.controls == "ice" or args.controls ==  "both":
        control2 = Control(control_ice_thickness_surf)

    
    control_ice_thickness.interpolate(control_ice_thickness_surf, allow_missing_dofs=True)
    
    ice_load = B_mu * rho_ice * Hice1 * control_ice_thickness 
    
#    ice_load = B_mu * rho_ice * (Hice1 * disc1 + Hice2 * disc2)

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
#        boundary.top: {'normal_stress': ice_load, 'free_surface': {}},
        boundary.top: {'free_surface': {'normal_stress': ice_load}},
    }

#    gd = GeodynamicalDiagnostics(z, density, boundary.bottom, boundary.top)
    # -


    # We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
    # needed for the viscoelastic loading problem.
    if args.burgers:
        shearmod_list = [0.5*shear_modulus, 0.5*shear_modulus]
        visc_list = [0.5*viscosity, burg_ratio*0.5*viscosity]
    else:
        shearmod_list = [shear_modulus]
        visc_list = [viscosity] 

    approximation = CompressibleInternalVariableApproximation(bulk_modulus=bulk_modulus, density=density, shear_modulus=shearmod_list, viscosity=visc_list, B_mu=B_mu, bulk_shear_ratio=args.bulk_shear_ratio)

    # We finally come to solving the variational problem, with solver
    # objects for the Stokes system created. We pass in the solution fields `z` and various fields
    # needed for the solve along with the approximation, timestep and boundary conditions.
    #

    direct_stokes_solver_parameters = {
        "snes_monitor": None,
        "snes_converged_reason": None,
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
                            "ksp_monitor": None,
                            "pc_type": "python",
                            "pc_python_type": "firedrake.AssembledPC",
                            "assembled_pc_type": "gamg",
                            "assembled_mg_levels_pc_type": "sor",
                            "assembled_pc_gamg_threshold": 0.01,
                            "assembled_pc_gamg_square_graph": 100,
                            "assembled_pc_gamg_coarse_eq_limit": 1000,
                            "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
                            }

    nullspace = rigid_body_modes(V, rotational=True)
    near_nullspace = rigid_body_modes(V, rotational=True, translations=[0, 1])

    solver = InternalVariableSolver(u, approximation, dt=dt, internal_variables=m_list, bcs=stokes_bcs,
                                            solver_parameters=iterative_parameters,
                                            nullspace=nullspace, transpose_nullspace=nullspace,
                                            near_nullspace=near_nullspace)


    # We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

    # +
    # Create output file
    OUTPUT = args.write_output
    vertical_displacement = Function(V.sub(1), name="radial displacement")  # Function to store vertical displacement for output

    if OUTPUT:
        output_file = VTKFile(f"{args.output_path}{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.pvd")
        output_file.write(u,*m_list, vertical_displacement)

    plog = ParameterLog(args.output_path+"params.log", mesh)
    plog.log_str(
        "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
    )
    
    velocity = Function(u, name="velocity")
    disp_old = Function(u, name="old_disp").assign(u)

    checkpoint_filename = f"{args.output_path}{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulktoshear{args.bulk_shear_ratio}-nondim-chk.h5"

    displacement_filename = f"{args.output_path}displacement-{name}-ncells{args.ncells}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.dat"

    # Initial displacement at time zero is zero
    displacement_min_array = [[0.0, 0.0]]

    def integrated_time_misfit(timestep, velocity_misfit, displacement_misfit):
        with CheckpointFile(checkpoint_file, 'r') as afile:
            target_displacement = afile.load_function(mesh, name="Displacement", idx=timestep)
            target_velocity = afile.load_function(mesh, name="Velocity", idx=timestep)
        circumference = 2 * pi * radius_values_tilde[0]
        velocity_error = velocity - target_velocity
        velocity_scale = 1e-5
        velocity_misfit += assemble(dot(velocity_error, velocity_error) / (circumference * velocity_scale**2) * ds(boundary.top))

        displacement_error = u - target_displacement
        displacement_scale = 1e-4
        displacement_misfit += assemble(dot(displacement_error, displacement_error) / (circumference * displacement_scale**2) * ds(boundary.top))
        return velocity_misfit, displacement_misfit
    
    def integrated_time_values(integrated_velocity, integrated_displacement, integrated_velocity_r, integrated_velocity_t, integrated_displacement_r, integrated_displacement_t):
        circumference = 2 * pi * radius_values_tilde[0]
        velocity_scale = 1e-5
        integrated_velocity += assemble(velocity**2 / (circumference * velocity_scale**2) * ds(boundary.top))
        
        vr = vc(velocity)
        vt = velocity - vr * k_vec
        integrated_velocity_r += assemble(vr**2 / (circumference * velocity_scale**2) * ds(boundary.top))
        integrated_velocity_t += assemble(vt**2 / (circumference * velocity_scale**2) * ds(boundary.top))
        
        displacement_scale = 1e-4
        disp_r = vc(u)
        disp_t =  u - disp_r * k_vec
        integrated_displacement += assemble(u**2 / (circumference * displacement_scale**2) * ds(boundary.top))
        integrated_displacement += assemble(disp_r**2 / (circumference * displacement_scale**2) * ds(boundary.top))
        integrated_displacement += assemble(disp_t**2 / (circumference * displacement_scale**2) * ds(boundary.top))
        return integrated_velocity, integrated_displacement, integrated_velocity_r, integrated_velocity_t, integrated_displacement_r, integrated_displacement_t
    
    velocity_misfit = 0
    displacement_misfit = 0
    integrated_velocity = 0
    integrated_velocity_r = 0
    integrated_velocity_t = 0
    integrated_displacement = 0
    integrated_displacement_r = 0
    integrated_displacement_t = 0

    # -

    # Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter.
    # At each step we call `solve` to calculate the incremental displacement and pressure fields. This
    # will update the displacement at the surface and stress values accounting for the time dependent
    # Maxwell consitutive equation.

    for timestep in range(1, max_timesteps+1):
        # update time first so that ice load begins
        time.assign(time+dt)
        solver.solve()
        
        velocity.interpolate((u - disp_old)/dt)
        disp_old.assign(u) 
        
        velocity_misfit, displacement_misfit = integrated_time_misfit(timestep, velocity_misfit, displacement_misfit)
        integrated_velocity, integrated_displacement, integrated_velocity_r, integrated_velocity_t, integrated_displacement_r, integrated_displacement_t = integrated_time_values(integrated_velocity, integrated_displacement, integrated_velocity_r, integrated_velocity_t, integrated_displacement_r, integrated_displacement_t)

        # Log diagnostics:
        # Compute diagnostics:
        # output dimensional vertical displacement
        vertical_displacement.interpolate(vc(u)*D)
        bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, boundary.top)
        displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
        displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
        log("Greatest (-ve) displacement", displacement_min)
        displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
        displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
        log("Greatest (+ve) displacement", displacement_max)
        displacement_min_array.append([float(characteristic_maxwell_time*time.dat.data[0]/year_in_seconds), displacement_min])

    #    disp_norm_L2surf = assemble((z.subfunctions[0][vertical_component])**2 * ds(boundary.top))
     #   log("L2 surface norm displacement", disp_norm_L2surf)

      #  disp_norm_L1surf = assemble(abs(z.subfunctions[0][vertical_component]) * ds(boundary.top))
       # log("L1 surface norm displacement", disp_norm_L1surf)

        #integrated_disp = assemble(z.subfunctions[0][vertical_component] * ds(boundary.top))
        #log("Integrated displacement", integrated_disp)

        if timestep % output_frequency == 0:
            log("timestep", timestep)

            if OUTPUT:
                output_file.write(u, *m_list, vertical_displacement)

            with CheckpointFile(checkpoint_filename, "w") as checkpoint:
                checkpoint.save_function(u, name="Stokes")
                for mi, m in enumerate(m_list):
                    checkpoint.save_function(m, name=f"m {mi}")

            if MPI.COMM_WORLD.rank == 0:
                np.savetxt(displacement_filename, displacement_min_array)

#            plog.log_str(f"{timestep} {float(time)} {float(dt)} "
#                         f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} "
#                         )

    circumference = 2 * pi * radius_values_tilde[0]
    area =  assemble(Constant(1) * dx(domain=mesh))

    ice_damping = args.ice_damping * assemble((control_ice_thickness- original_ice_thickness) ** 2 / circumference  * ds(boundary.top))
    ice_smoothing = args.ice_smoothing * assemble(dot(grad(control_ice_thickness - original_ice_thickness), grad(control_ice_thickness - original_ice_thickness)) / circumference * ds(boundary.top))
    
    visc_damping = args.visc_damping * assemble((control_viscosity -original_viscosity) ** 2 / area  * dx)
    visc_smoothing = args.visc_smoothing * assemble(dot(grad(control_viscosity-original_viscosity), grad(control_viscosity-original_viscosity)) / area * dx)
    
    if args.time_integrated_sensitivity:
        if args.obj_radial:
            objective = (integrated_displacement_r + integrated_velocity_r) / max_timesteps
        elif args.obj_tangential:
            objective = (integrated_displacement_t + integrated_velocity_t) / max_timesteps
        else:
            objective = (integrated_displacement + integrated_velocity) / max_timesteps
    elif args.final_sensitivity:
        final_integrated_vel = 0
        final_integrated_vel_r = 0
        final_integrated_vel_t = 0
        final_integrated_disp = 0
        final_integrated_disp_r = 0
        final_integrated_disp_t = 0
        final_integrated_vel, final_integrated_disp, final_integrated_vel_r, final_integrated_vel_t, final_integrated_disp_r, final_integrated_disp_t = integrated_time_values(final_integrated_vel, final_integrated_disp, final_integrated_vel_r, final_integrated_vel_t, final_integrated_disp_r, final_integrated_disp_t)
        if args.obj_radial:
            objective = (final_integrated_disp_r + final_integrated_vel_r)
        elif args.obj_tangential:
            objective = (final_integrated_disp_t + final_integrated_vel_t) 
        else:
            objective = (final_integrated_disp + final_integrated_vel) 
    else:
        # misfit functional
        objective = (displacement_misfit + velocity_misfit) / max_timesteps 
        objective += ice_damping + ice_smoothing
        objective += visc_damping + visc_smoothing

    log("J = ", objective)
    
    
    # calculate ice error cf target
    ice_error = control_ice_thickness - target_normalised_ice_thickness
    ice_error_L2 = assemble(dot(ice_error, ice_error) / (circumference * dot(target_normalised_ice_thickness, target_normalised_ice_thickness) + 1e-16 ) * ds(boundary.top))
    
    # calculate viscosity error cf target
    target_log_viscosity = ln(target_viscosity/background_viscosity) / ln(10)
    visc_error = control_viscosity - target_log_viscosity
    visc_error_L2 = assemble(dot(visc_error, visc_error) / (area * dot(target_log_viscosity, target_log_viscosity) + 1e-16 ) * dx)
   
    
    pause_annotation()
    
    # storing adjoint results
    updated_ice_thickness = Function(control_ice_thickness, name="updated ice thickness")
    updated_viscosity = Function(target_viscosity, name="updated viscosity")
    updated_log_viscosity = Function(control_viscosity, name="updated control viscosity")
    updated_solution_file = VTKFile(f"{args.output_path}{name}_sol.pvd")
    updated_displacement = Function(u, name="updated displacement")
    updated_velocity = Function(u, name="updated velocity")
    updated_out_file = VTKFile(f"{args.output_path}{name}_updated_out.pvd")
    
    updated_solution_file = VTKFile(f"{args.output_path}{name}_sol.pvd")
    updated_out_file = VTKFile(f"{args.output_path}{name}_updated_out.pvd")

    controls_checkpoint_filename = f"{args.output_path}{name}_controls.h5"

    functional_values = []
    
    with CheckpointFile(checkpoint_file, 'r') as afile:
        final_target_displacement = afile.load_function(mesh, name="Displacement", idx=max_timesteps)
        final_target_velocity = afile.load_function(mesh, name="Velocity", idx=max_timesteps)
    
    # surface displacement outputs
    surface_displacement_filename = f"{args.output_path}{name}_surface_final_disp.csv"
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
    
    # surface ice outputs
    surface_ice_filename = f"{args.output_path}{name}_surface_ice.csv"
    f_ice = Function(P1_vec).interpolate(as_vector([X[0], X[1]]))
    bc_ice = DirichletBC(control_ice_thickness.function_space(), 0, boundary.top)

    surface_x_ice = f_ice.sub(0).dat.data_ro_with_halos[bc_ice.nodes]
    surface_x_all_ice = f_ice.sub(0).comm.gather(surface_x_ice)
    surface_y_ice = f_ice.sub(1).dat.data_ro_with_halos[bc_ice.nodes]
    surface_y_all_ice = f_ice.sub(1).comm.gather(surface_y_ice)
    ice_df = pd.DataFrame()

    if MPI.COMM_WORLD.rank == 0:
        surface_x_concat_ice = np.concatenate(surface_x_all_ice)
        ice_df['surface_x'] = surface_x_concat_ice
        surface_y_concat_ice = np.concatenate(surface_y_all_ice)
        ice_df['surface_y'] = surface_y_concat_ice
    
    class eval_cb_class(object):
        def __init__(self):
            self.counter = 0
            self.functional_values = []
            self.ice_misfit = []
            self.viscosity_misfit = []
            self.ice_change = []
            self.viscosity_change = []
            self.ice_change_fromlog = []
            self.viscosity_change_fromlog = []


        def __call__(self, J, m):
            if self.functional_values:
                self.functional_values.append(min(J, min(self.functional_values)))
            else:
                self.functional_values.append(J)

            log("displacement misfit", displacement_misfit.block_variable.checkpoint / max_timesteps)
            log("velocity misfit", velocity_misfit.block_variable.checkpoint / max_timesteps)
            log("ice smoothing", ice_smoothing.block_variable.checkpoint)
            log("ice damping", ice_damping.block_variable.checkpoint)
            log("viscosity smoothing", visc_smoothing.block_variable.checkpoint)
            log("viscosity damping", visc_damping.block_variable.checkpoint)


            
            # Write out values of control and final forward model results
            if args.controls == "ice" or args.controls =="both":
                updated_ice_thickness.assign(control_ice_thickness.block_variable.checkpoint)

            
            # Write out values of control and final forward model results
            updated_viscosity.interpolate(background_viscosity * 10**control_viscosity.block_variable.checkpoint)
            updated_log_viscosity.assign(control_viscosity.block_variable.checkpoint)
            updated_displacement.interpolate(u.block_variable.checkpoint)
            updated_velocity.interpolate(velocity.block_variable.checkpoint)
            updated_solution_file.write(updated_ice_thickness, target_normalised_ice_thickness, updated_viscosity, 
                    target_viscosity, updated_displacement, final_target_displacement, updated_velocity, final_target_velocity)
            updated_out_file.write(updated_displacement, final_target_displacement)

            with CheckpointFile(controls_checkpoint_filename, "w") as checkpoint:
                checkpoint.save_function(updated_log_viscosity, name="control viscosity")
                checkpoint.save_function(updated_ice_thickness, name="control normalised ice thickness")
            
            if args.controls =="ice" or args.controls =="both":
                # calculate ice error cf target
                self.ice_misfit.append(ice_error_L2.block_variable.checkpoint)
                log("ice error", ice_error_L2.block_variable.checkpoint)
            if args.controls == "viscosity" or args.controls == "both":
                # calculate viscosity error cf target
                self.viscosity_misfit.append(visc_error_L2.block_variable.checkpoint)
                log("viscosity error", visc_error_L2.block_variable.checkpoint)
                if args.burgers:
                    log("burgers ratio", burg_ratio.block_variable.checkpoint)
            
            # write out surface displacement 
            disp_x.interpolate(u.block_variable.checkpoint[0]*D)
            surface_disp_x = disp_x.dat.data_ro_with_halos[bc_displacement.nodes]
            surface_disp_x_all = disp_x.comm.gather(surface_disp_x)
            disp_y.interpolate(u.block_variable.checkpoint[1]*D)
            surface_disp_y = disp_y.dat.data_ro_with_halos[bc_displacement.nodes]
            surface_disp_y_all = disp_y.comm.gather(surface_disp_y)

            if MPI.COMM_WORLD.rank == 0:
                surface_disp_x_concat = np.concatenate(surface_disp_x_all)
                displacement_df[f'surface_disp_x_step{self.counter}'] = surface_disp_x_concat
                
                surface_disp_y_concat = np.concatenate(surface_disp_y_all)
                displacement_df[f'surface_disp_y_step{self.counter}'] = surface_disp_y_concat
            
                displacement_df.to_csv(surface_displacement_filename)
            
            # write out surface ice 
            if self.counter ==0:
                # write out target ice on first iteration
                surface_ice_target = target_normalised_ice_thickness.dat.data_ro_with_halos[bc_ice.nodes]
                surface_ice_all_target = target_normalised_ice_thickness.comm.gather(surface_ice_target)

                if MPI.COMM_WORLD.rank == 0:
                    surface_ice_concat_target = np.concatenate(surface_ice_all_target)
                    ice_df[f'surface_ice_target'] = surface_ice_concat_target

            # updated surface ice
            surface_ice = updated_ice_thickness.dat.data_ro_with_halos[bc_ice.nodes]
            surface_ice_all = updated_ice_thickness.comm.gather(surface_ice)

            if MPI.COMM_WORLD.rank == 0:
                surface_ice_concat = np.concatenate(surface_ice_all)
                ice_df[f'surface_ice_step{self.counter}'] = surface_ice_concat
                ice_df.to_csv(surface_ice_filename)
            
            if MPI.COMM_WORLD.rank == 0:
                with open(f"{args.output_path}{name}_functional.txt", "w") as f:
                    f.write("\n".join(str(x) for x in self.functional_values))
                if args.controls =="ice" or args.controls == "both":
                    with open(f"{args.output_path}{name}_ice_misfit.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.ice_misfit))
                    with open(f"{args.output_path}{name}_ice_change.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.ice_change))
                    with open(f"{args.output_path}{name}_ice_change_fromlog.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.ice_change_fromlog))
                if args.controls =="viscosity" or args.controls == "both":
                    with open(f"{args.output_path}{name}_viscosity_misfit.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.viscosity_misfit))
                    with open(f"{args.output_path}{name}_viscosity_change.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.viscosity_change))
                    with open(f"{args.output_path}{name}_viscosity_change_fromlog.txt", "w") as f:
                        f.write("\n".join(str(x) for x in self.viscosity_change_fromlog))

            self.counter += 1
        
    eval_cb = eval_cb_class()
    ice_thickness_lb = Function(control_ice_thickness_surf.function_space(), name="Lower bound ice thickness")
    ice_thickness_ub = Function(control_ice_thickness_surf.function_space(), name="Upper bound ice thickness")
    ice_thickness_lb.assign(0.0)
    ice_thickness_ub.assign(5)

    ice_bounds = [ice_thickness_lb, ice_thickness_ub]
    
    viscosity_lb = Function(control_viscosity.function_space(), name="Lower bound ice thickness")
    viscosity_ub = Function(control_viscosity.function_space(), name="Upper bound ice thickness")
    viscosity_lb.assign(-3)
    viscosity_ub.assign(6)

    viscosity_bounds = [viscosity_lb, viscosity_ub]
    
    burg_ratio_lb = Function(R).assign(1e-3)
    burg_ratio_ub = Function(R).assign(100)
    burg_ratio_bounds = [burg_ratio_lb, burg_ratio_ub]

    if args.controls =="ice":
        bounds = ice_bounds
    elif args.controls =="viscosity":
        bounds = viscosity_bounds
        if args.burgers:
            bounds = [viscosity_bounds, burg_ratio_bounds]
    else:
        bounds = [viscosity_bounds, ice_bounds]
        if args.burgers:
            bounds.append(burg_ratio_bounds)
    inverse_problem = {}

    if args.controls =="ice":
        clist = [control_ice_thickness_surf]
        c = [control2]
    elif args.controls =="viscosity":
        clist = [control_viscosity]
        c = [control1]
        if args.burgers:
            clist.append(burg_ratio)
            c.append(control3)
    else:
        clist = [control_viscosity, control_ice_thickness]
        c = [control1, control2]
        if args.burgers:
            clist.append(burg_ratio)
            c.append(control3)
    # Keep track of what the control function is
    inverse_problem["control"] = clist

    # The ReducedFunctional that is to be minimised
    inverse_problem["reduced_functional"] = ReducedFunctional(objective, c, eval_cb_post=eval_cb)
    inverse_problem["bounds"] = bounds


    return inverse_problem

inverse()
