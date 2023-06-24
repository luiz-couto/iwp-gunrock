#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/container/vector.hxx>
#include <thrust/for_each.h>
#include <cstdio>
#include <map>
#include <algorithm> // std::min
#include <cmath>

#define log(msg, x) std::cout << msg << ": " << #x << " = " << x << std::endl;
#define debug(x) std::cout << #x << " = " << x << std::endl;
#define debug2(x, y) std::cout << #x << " = " << x << " --- " << #y << " = " << y << "\n";
#define debugLine(i) std::cout << "PASSOU AQUIIII" \
                               << " --- " << i << std::endl;

__host__ __device__ constexpr int euclideanDistance(int v1, int v2, int img_width)
{
    double x1 = (double)(v1 % img_width);
    double y1 = (double)(v1 / img_width);

    double x2 = (double)(v2 % img_width);
    double y2 = (double)(v2 / img_width);

    double x = x1 - x2; // calculating number to square in next step
    double y = y1 - y2;
    double dist = sqrt(pow(x, 2) + pow(y, 2));

    return round(dist);
}

namespace iwp
{

    namespace dist
    {

        template <typename vertex_t>
        struct param_t
        {
            vertex_t *bin_img;
            int img_width;
            int img_height;
            param_t(vertex_t *_bin_img,
                    int _img_width,
                    int _img_height) : bin_img(_bin_img), img_width(_img_width), img_height(_img_height) {}
        };

        template <typename vertex_t>
        struct result_t
        {
            vertex_t *vr_diagram;
            result_t(vertex_t *_vr_diagram) : vr_diagram(_vr_diagram) {}
        };

        template <typename graph_t, typename param_type, typename result_type>
        struct problem_t : gunrock::problem_t<graph_t>
        {
            param_type param;
            result_type result;

            problem_t(graph_t &G,
                      param_type &_param,
                      result_type &_result,
                      std::shared_ptr<gunrock::gcuda::multi_context_t> _context)
                : gunrock::problem_t<graph_t>(G, _context),
                  param(_param),
                  result(_result) {}

            void init() override {}

            void reset() override {}
        };

        template <typename problem_t>
        struct enactor_t : gunrock::enactor_t<problem_t>
        {
            enactor_t(problem_t *_problem,
                      std::shared_ptr<gunrock::gcuda::multi_context_t> _context)
                : gunrock::enactor_t<problem_t>(_problem, _context) {}

            using vertex_t = typename problem_t::vertex_t;
            using edge_t = typename problem_t::edge_t;
            using weight_t = typename problem_t::weight_t;
            using frontier_t = typename enactor_t<problem_t>::frontier_t;

            void prepare_frontier(frontier_t *f, gunrock::gcuda::multi_context_t &context) override
            {
                debugLine("Prepare Frontier");

                auto P = this->get_problem();
                auto G = P->get_graph();

                vertex_t *vr_diagram = P->result.vr_diagram;
                vertex_t *bin_img = P->param.bin_img;
                int width = P->param.img_width;
                int height = P->param.img_height;

                thrust::device_vector<vertex_t> device_frontier(width * height, -1);
                vertex_t *f_pointer = device_frontier.data().get();

                auto fill_frontier = [G, vr_diagram, bin_img, width, height, f_pointer] __device__(vertex_t const &v)
                {
                    int BG = 0;
                    int FR = 1;

                    if (bin_img[v] == BG)
                    {
                        vr_diagram[v] = v;

                        edge_t startEdge = G.get_starting_edge(v);
                        auto numberNgbs = G.get_number_of_neighbors(v);

                        for (edge_t e = startEdge; e < startEdge + numberNgbs; e++)
                        {
                            vertex_t ngb = G.get_destination_vertex(e);
                            if (bin_img[ngb] == FR)
                            {
                                f_pointer[v] = v;
                                break;
                            }
                        }
                    }
                };

                thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0),
                                 thrust::make_counting_iterator<vertex_t>(G.get_number_of_vertices()),
                                 fill_frontier);

                cudaDeviceSynchronize();

                int result = thrust::count(thrust::device, device_frontier.begin(), device_frontier.end(), -1);

                thrust::remove(device_frontier.begin(), device_frontier.end(), -1);
                device_frontier.resize((width * height) - result);

                if (f->get_capacity() < device_frontier.size())
                    f->reserve(device_frontier.size());

                // Set the new number of elements.
                f->set_number_of_elements(device_frontier.size());
                thrust::copy(device_frontier.begin(), device_frontier.end(), f->begin());

                debug(f->get_number_of_elements());
            }

            void loop(gunrock::gcuda::multi_context_t &context) override
            {
                auto E = this->get_enactor();
                auto P = this->get_problem();
                auto G = P->get_graph();

                vertex_t *vr_diagram = P->result.vr_diagram;
                vertex_t *bin_img = P->param.bin_img;
                int img_width = P->param.img_width;

                auto advance_op = [vr_diagram, bin_img, img_width] __host__ __device__(
                                      vertex_t const &source,   // ... source
                                      vertex_t const &neighbor, // neighbor
                                      edge_t const &edge,       // edge
                                      weight_t const &weight    // weight (tuple).
                                      ) -> bool
                {
                    if (euclideanDistance(neighbor, vr_diagram[source], img_width) < euclideanDistance(neighbor, vr_diagram[neighbor], img_width))
                    {
                        vr_diagram[neighbor] = vr_diagram[source];
                        return true;
                    }
                    return false;
                };

                gunrock::operators::advance::execute<gunrock::operators::load_balance_t::thread_mapped>(G, E, advance_op, context);
            }

        }; // struct enactor_t

        template <typename graph_t>
        float run(graph_t &G,
                  typename graph_t::vertex_type *bin_img,    // Parameter
                  const int img_width,                       // Parameter
                  const int img_height,                      // Parameter
                  typename graph_t::vertex_type *vr_diagram, // Output
                  std::shared_ptr<gunrock::gcuda::multi_context_t> context =
                      std::shared_ptr<gunrock::gcuda::multi_context_t>(
                          new gunrock::gcuda::multi_context_t(0)) // Context
        )
        {
            // <user-defined>

            using vertex_t = typename graph_t::vertex_type;

            using param_type = param_t<vertex_t>;
            using result_type = result_t<vertex_t>;

            param_type param(bin_img, img_width, img_height);
            result_type result(vr_diagram);
            // </user-defined>

            using problem_type = problem_t<graph_t, param_type, result_type>;
            using enactor_type = enactor_t<problem_type>;

            problem_type problem(G, param, result, context);
            problem.init();
            problem.reset();

            enactor_type enactor(&problem, context);
            return enactor.enact();
            // </boiler-plate>
        }
    }
}