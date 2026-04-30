/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2005 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
namespace Step20
{
  using namespace dealii;

  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem(const unsigned int degree);
    void run();

  private:
    void make_grid_and_dofs();
    void assemble_system();
    void solve();
    void compute_errors() const;
    void output_results() const;

    const unsigned int degree;

    Triangulation<dim>  triangulation;
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;

    // BlockSparsityPattern      sparsity_pattern;
    // BlockSparseMatrix<double> system_matrix;

    // BlockVector<double> solution;
    // BlockVector<double> system_rhs;
    std::vector<IndexSet> partitioning;

    TrilinosWrappers::BlockSparseMatrix system_matrix;

    TrilinosWrappers::MPI::BlockVector solution;
    TrilinosWrappers::MPI::BlockVector system_rhs;
  };

  namespace PrescribedSolution
  {
    constexpr double alpha = 0.3;
    constexpr double beta  = 1;

    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
    };

    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
      PressureBoundaryValues()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
    };

    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
      ExactSolution()
        : Function<dim>(dim + 1)
      {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double>   &value) const override;
    };

    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                     const unsigned int /*component*/) const
    {
      return 0;
    }

    template <int dim>
    double
    PressureBoundaryValues<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
    {
      return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
               alpha * p[0] * p[0] * p[0] / 6);
    }

    template <int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double>   &values) const
    {
      AssertDimension(values.size(), dim + 1);

      values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
      values(1) = alpha * p[0] * p[1];
      values(2) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                    alpha * p[0] * p[0] * p[0] / 6);
    }

    template <int dim>
    class KInverse : public TensorFunction<2, dim>
    {
    public:
      KInverse()
        : TensorFunction<2, dim>()
      {}

      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<2, dim>>   &values) const override;
    };

    template <int dim>
    void KInverse<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<Tensor<2, dim>>   &values) const
    {
      (void)points;
      AssertDimension(points.size(), values.size());

      for (auto &value : values)
        value = unit_symmetric_tensor<dim>();
    }
  } // namespace PrescribedSolution

  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem(const unsigned int degree)
    : degree(degree)
    , fe(FE_RaviartThomas<dim>(degree), FE_DGQ<dim>(degree))
    , dof_handler(triangulation)
  {}

  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(5);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);

    const unsigned int n_u = dofs_per_component[0];
    const unsigned int n_p = dofs_per_component[dim];

    std::vector<types::global_dof_index> block_sizes = {n_u, n_p};

    partitioning.resize(2);
    partitioning[0] = complete_index_set(n_u);
    partitioning[1] = complete_index_set(n_p);

    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    system_matrix.reinit(dsp);

    solution.reinit(partitioning, MPI_COMM_WORLD);
    system_rhs.reinit(partitioning, MPI_COMM_WORLD);
  }

  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system()
  {
    const QGauss<dim>     quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 2);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const PrescribedSolution::RightHandSide<dim> right_hand_side;
    const PrescribedSolution::PressureBoundaryValues<dim>
                                            pressure_boundary_values;
    const PrescribedSolution::KInverse<dim> k_inverse;

    std::vector<double>         rhs_values(n_q_points);
    std::vector<double>         boundary_values(n_face_q_points);
    std::vector<Tensor<2, dim>> k_inverse_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);
        k_inverse.value_list(fe_values.get_quadrature_points(),
                             k_inverse_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
              const double div_phi_i_u = fe_values[velocities].divergence(i, q);
              const double phi_i_p     = fe_values[pressure].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_u =
                    fe_values[velocities].value(j, q);
                  const double div_phi_j_u =
                    fe_values[velocities].divergence(j, q);
                  const double phi_j_p = fe_values[pressure].value(j, q);

                  local_matrix(i, j) +=
                    (phi_i_u * k_inverse_values[q] * phi_j_u 
                     - phi_i_p * div_phi_j_u                 
                     - div_phi_i_u * phi_j_p)                
                    * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p * rhs_values[q] * fe_values.JxW(q);
            }

        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              fe_face_values.reinit(cell, face);

              pressure_boundary_values.value_list(
                fe_face_values.get_quadrature_points(), boundary_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  local_rhs(i) += -(fe_face_values[velocities].value(i, q) * 
                                    fe_face_values.normal_vector(q) *        
                                    boundary_values[q] *                     
                                    fe_face_values.JxW(q));
            }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);

        // Use compress_add for Trilinos matrices
        // system_matrix.add(local_dof_indices, local_matrix);
        // system_rhs.add(local_dof_indices, local_rhs);
      }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  // template <int dim>
  // void MixedLaplaceProblem<dim>::solve()
  // {
  // SolverControl solver_control(5000, 1e-10);
  // TrilinosWrappers::SolverGMRES solver(solver_control);  // It will not working with full matrix as we have blocksparse matrix
  // PreconditionIdentity preconditioner;
  
  // solver.solve(system_matrix, solution, system_rhs, preconditioner);
  
  // std::cout << solver_control.last_step()
  //           << " GMRES iterations to obtain convergence." << std::endl;
  // }

  template <int dim>
  void MixedLaplaceProblem<dim>::solve()
  {
  const auto &M = system_matrix.block(0, 0);
  const auto &B = system_matrix.block(0, 1);
  const auto &F = system_rhs.block(0);
  const auto &G = system_rhs.block(1);
  auto &U = solution.block(0);
  auto &P = solution.block(1);

  // Create linear operators
  const auto op_M = linear_operator<TrilinosWrappers::MPI::Vector>(M);
  const auto op_B = linear_operator<TrilinosWrappers::MPI::Vector>(B);

  // Use deal.II's CG solver with Trilinos vectors
  ReductionControl reduction_control_M(2000, 1.0e-18, 1.0e-10);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_M(reduction_control_M);
  
  TrilinosWrappers::PreconditionJacobi preconditioner_M;
  preconditioner_M.initialize(M);
  
  const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);
  
  // Approximate Schur complement preconditioner
  const auto op_aS = transpose_operator(op_B) * op_M_inv * op_B;
  
  IterationNumberControl iteration_number_control_aS(30, 1.e-18);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_aS(iteration_number_control_aS);
  const auto preconditioner_S = inverse_operator(op_aS, solver_aS, PreconditionIdentity());
  
  // Actual Schur complement
  const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
  const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;
  
  // Solve for pressure
  SolverControl solver_control_S(2000, 1.e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
  const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
  P = op_S_inv * schur_rhs;
  
  // Output convergence info
  std::cout << solver_control_S.last_step()
              << " CG Schur complement (S) iterations to obtain convergence."
              << std::endl;
  
  // Solve for velocities
  U = op_M_inv * (F - op_B * P);
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors() const
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);

    PrescribedSolution::ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors(triangulation.n_active_cells());

    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);

    // Convert Trilinos solution to deal.II Vector for error computation
    Vector<double> solution_dealii(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      solution_dealii(i) = solution(i);

    VectorTools::integrate_difference(dof_handler,
                                      solution_dealii,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution_dealii,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
  }

  template <int dim>
  void MixedLaplaceProblem<dim>::output_results() const
  {
    std::vector<std::string> solution_names(dim, "u");
    solution_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // Convert Trilinos solution to deal.II Vector for output
    Vector<double> solution_dealii(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      solution_dealii(i) = solution(i);

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution_dealii,
                             solution_names,
                             interpretation);

    data_out.build_patches(degree + 1);

    std::ofstream output("solution.vtu");
    data_out.write_vtu(output);
  }


  template <int dim>
  void MixedLaplaceProblem<dim>::run()
  {
    make_grid_and_dofs();
    assemble_system();
    solve();
    compute_errors();
    output_results();
  }
} 

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      // Initialize MPI (required for Trilinos even if running sequentially)
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                  ExcMessage("This program can only be run in serial, use mpirun -np 1 ./step_20"));

      const unsigned int fe_degree = 0;
      MixedLaplaceProblem<2> mixed_laplace_problem(fe_degree);
      mixed_laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
