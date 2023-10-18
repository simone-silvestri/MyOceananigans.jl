using Oceananigans
using Oceananigans.Operators

using Oceananigans.Architectures: device, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.Models.NonhydrostaticModels: PressureSolver, calculate_pressure_source_term_fft_based_solver!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using Oceananigans.DistributedComputations: ranks, topology, halo_size

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!
import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

struct PCGPoissonSolver{R, G, I, K, S}
    rhs :: R
    grid :: G
    localiter :: Int
    iter :: I
    kernel_params :: K
    pcg_solver :: S
end

const DDPC = DiagonallyDominantPreconditioner

function PCGPoissonSolver(grid;
                          preconditioner = DDPC(),
                          localiter = 20,
                          reltol = eps(eltype(grid)),
                          abstol = 0,
                          kw...)

    grid = pcg_inflate_halo_size(localiter, grid)
    rhs  = CenterField(grid)

    kernel_size    = pcg_augmented_kernel_size(grid)
    kernel_offsets = pcg_augmented_kernel_offsets(grid)
    kernel_params  = KernelParameters(kernel_size, kernel_offsets)

    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!; reltol, abstol,
                                                       preconditioner,
                                                       template_field = rhs,
                                                       kw...)

    return PCGPoissonSolver(rhs, grid, localiter, Ref(0), kernel_params, pcg_solver)
end

@inline function pcg_augmented_kernel_size(grid::DistributedGrid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 ? Nx : (Tx == RightConnected || Tx == LeftConnected ? Nx + Hx - 1 : Nx + 2Hx - 2)
    Ay = Ry == 1 ? Ny : (Ty == RightConnected || Ty == LeftConnected ? Ny + Hy - 1 : Ny + 2Hy - 2)

    return (Ax, Ay, Nz)
end
   
@inline function pcg_augmented_kernel_offsets(grid::DistributedGrid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    Rx, Ry, _ = architecture(grid).ranks

    Ax = Rx == 1 || Tx == RightConnected ? 0 : - Hx + 1
    Ay = Ry == 1 || Ty == RightConnected ? 0 : - Hy + 1

    return (Ax, Ay, 0)
end

@inline pcg_inflate_halo_size(localiter, grid) = grid

@inline function pcg_inflate_halo_size(localiter, grid::DistributedGrid)
    Hx, Hy, Hz = halo_size(grid)
    Rx, Ry, _  = ranks(architecture(grid).partition)

    Ax = Rx == 1 ? Hx : localiter + 1
    Ay = Ry == 1 ? Hy : localiter + 1

    return (Ax, Ay, Hz)
end

@kernel function calculate_pressure_source_term!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@inline laplacianᶜᶜᶜ(i, j, k, grid, ϕ) = ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = laplacianᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ, params, localiter)
    grid = ϕ.grid
    arch = architecture(grid)

    only_local_halos = only_local_halo_iter(iter, localiter)
    fill_halo_regions!(ϕ; only_local_halos)

    launch!(arch, grid, params, laplacian!, ∇²ϕ, grid, ϕ)

    iter[] = only_local_halos ? 0 : iter[] + 1

    return nothing
end

@inline only_local_halo_iter(iter, ::Nothing) = false
@inline only_local_halo_iter(iter, localiter) = iter[] > localiter ? false : true

function solve_for_pressure!(pressure, solver::PCGPoissonSolver, Δt, U★)
    # TODO: Is this the right criteria?
    min_Δt = eps(typeof(Δt))
    Δt <= min_Δt && return pressure

    rhs = solver.rhs
    grid = solver.grid
    arch = architecture(grid)

    if grid isa ImmersedBoundaryGrid
        underlying_grid = grid.underlying_grid
    else
        underlying_grid = grid
    end

    launch!(arch, grid, :xyz, calculate_pressure_source_term!,
            rhs, underlying_grid, Δt, U★)

    mask_immersed_field!(rhs, zero(grid))
    fill_halo_regions!(rhs)

    # Solve pressure Pressure equation for pressure, given rhs
    # @info "Δt before pressure solve: $(Δt)"
    solve!(pressure, solver.pcg_solver, rhs, solver.kernel_params, solver.iter)

    solver.iter[] = 0

    return pressure
end

struct DiagonallyDominantPreconditioner end

@inline function precondition!(P_r, ::DiagonallyDominantPreconditioner, r, params, iter, args...)
    grid = r.grid
    arch = architecture(P_r)

    only_local_halos = only_local_halo_iter(iter, localiter)
    fill_halo_regions!(r; only_local_halos)

    launch!(arch, grid, params, _MITgcm_precondition!,
            P_r, grid, r)

    return P_r
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i, j, k, grid) / Δxᶠᶜᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j, k, grid) / Δyᶜᶠᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k, grid) / Δzᶜᶜᶠ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) / Δxᶠᶜᶜ(i+1, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) / Δyᶜᶠᶜ(i, j+1, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) / Δzᶜᶜᶠ(i, j, k+1, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - (Ax⁻(i, j, k, grid) +
                               Ax⁺(i, j, k, grid) +
                               Ay⁻(i, j, k, grid) +
                               Ay⁺(i, j, k, grid) +
                               Az⁻(i, j, k, grid) +
                               Az⁺(i, j, k, grid))

@inline heuristic_inverse_times_residuals(i, j, k, r, grid) =
    @inbounds 1 / Ac(i, j, k, grid) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
                                                    2 * Ax⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i+1, j, k, grid)) * r[i+1, j, k] -
                                                    2 * Ay⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j-1, k, grid)) * r[i, j-1, k] -
                                                    2 * Ay⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j+1, k, grid)) * r[i, j+1, k] -
                                                    2 * Az⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k-1, grid)) * r[i, j, k-1] -
                                                    2 * Az⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k+1, grid)) * r[i, j, k+1])

@kernel function _MITgcm_precondition!(P_r, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds P_r[i, j, k] = heuristic_inverse_times_residuals(i, j, k, r, grid)
end