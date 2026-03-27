#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <iostream>

using namespace dealii;

int
main()
{
  // Triangulation.
  Triangulation<2> tria;

  Point<2> center(0, 0);

  double inner_radius = 0.5, outer_radius = 1;

  GridGenerator::hyper_shell(tria, center, inner_radius, outer_radius, 5);

  // Refinement.
  for (unsigned step = 0; step < 3; ++step)
    {
      for (auto &cell : tria.active_cell_iterators())
        {
          for (auto v : cell->vertex_indices())
            {
              Point<2> vertex = cell->vertex(v);

              const double distance_from_center = vertex.distance(center);

              if (std::fabs(distance_from_center - inner_radius) <
                  1e-6 * inner_radius)
                {
                  cell->set_refine_flag();
                  break;
                }
            }
        }
      tria.execute_coarsening_and_refinement();
    }

  // Output mesh.
  GridOut grid_out;
  grid_out.write_msh(tria, "first_mesh.msh");
  
  return 0;
}
