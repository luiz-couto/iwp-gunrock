#pragma once
#include <opencv2/core.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/container/vector.hxx>
#include <thrust/for_each.h>
#include <cstdio>
#include <map>

#define debug(x) std::cout << #x << " = " << x << std::endl;
#define debug2(x, y) std::cout << #x << " = " << x << " --- " << #y << " = " << y << "\n";
#define debugLine(i) std::cout << "PASSOU AQUIIII" \
                               << " --- " << i << std::endl;

using pixel_coords = std::pair<int, int>;

template <typename S>
std::ostream &operator<<(std::ostream &os, const std::vector<S> &vector)
{
    for (auto element : vector)
    {
        os << element << " ";
    }
    return os;
}

namespace iwp
{
    int get1DCoords(cv::Mat &img, pixel_coords coords);
    pixel_coords get2DCoords(int width, int coord);
    std::vector<int> getPixelNeighbours(cv::Mat &img, pixel_coords coords);

    template <typename vertex_t, typename edge_t, typename weight_t>
    auto convertImgToGraph(cv::Mat &marker, cv::Mat &mask, vertex_t *markerValues, vertex_t *maskValues);

    float runMorphRec(cv::Mat &marker, cv::Mat &mask);

    template <typename vertex_t>
    struct param_t
    {
        vertex_t *mask;
        param_t(vertex_t *_mask) : mask(_mask) {}
    };

    template <typename vertex_t>
    struct result_t
    {
        vertex_t *marker;
        result_t(vertex_t *_marker) : marker(_marker) {}
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
            vertex_t *marker = P->result.marker;
            vertex_t *mask = P->param.mask;

            thrust::device_vector<vertex_t> device_frontier(100, -1);
            vertex_t *f_pointer = device_frontier.data().get();

            auto update_pixel = [G, marker, mask] __device__(vertex_t const &v)
            {
                auto startEdge = G.get_starting_edge(v);
                auto numberNgbs = G.get_number_of_neighbors(v);
                vertex_t greater = marker[v];

                for (auto e = startEdge; e < startEdge + numberNgbs; e++)
                {
                    vertex_t ngb = G.get_destination_vertex(e);
                    if (marker[ngb] > greater)
                        greater = marker[ngb];
                }

                if (greater > mask[v])
                    greater = mask[v];

                marker[v] = greater;
            };

            auto fill_frontier = [G, marker, mask, f_pointer, update_pixel] __device__(vertex_t const &v)
            {
                update_pixel(v);

                edge_t startEdge = G.get_starting_edge(v);
                auto numberNgbs = G.get_number_of_neighbors(v);

                for (edge_t e = startEdge; e < startEdge + numberNgbs; e++)
                {
                    vertex_t ngb = G.get_destination_vertex(e);
                    if ((marker[ngb] < marker[v]) && (marker[ngb] < mask[ngb]))
                    {
                        f_pointer[v] = ngb;
                    }
                }
            };

            auto policy = context.get_context(0)->execution_policy();

            // For each (count from 0...#_of_Vertices), and perform
            // the operation called update_pixel.
            thrust::for_each(policy,
                             thrust::make_counting_iterator<vertex_t>(0),                          // Begin: 0
                             thrust::make_counting_iterator<vertex_t>(G.get_number_of_vertices()), // End: # of Vertices
                             update_pixel                                                          // Unary operation
            );

            // DISCOVER A WAY TO DO IT IN A ANTI-RASTER MANNER
            thrust::for_each(policy,
                             thrust::make_counting_iterator<vertex_t>(0),                          // Begin: 0
                             thrust::make_counting_iterator<vertex_t>(G.get_number_of_vertices()), // End: # of Vertices
                             fill_frontier                                                         // Unary operation
            );

            thrust::host_vector<vertex_t> host_frontier = device_frontier;
            std::map<int, bool> entered_pixels;
            for (int i = 0; i < 100; i++)
            {
                if (host_frontier[i] != -1 && entered_pixels.find(host_frontier[i]) == entered_pixels.end())
                {
                    f->push_back(host_frontier[i]);
                    entered_pixels[host_frontier[i]] = true;
                }
            }

            f->print();
        }

        void loop(gunrock::gcuda::multi_context_t &context) override
        {
        }

    }; // struct enactor_t

    template <typename graph_t>
    float run(graph_t &G,
              typename graph_t::vertex_type *mask,   // Parameter
              typename graph_t::vertex_type *marker, // Output
              std::shared_ptr<gunrock::gcuda::multi_context_t> context =
                  std::shared_ptr<gunrock::gcuda::multi_context_t>(
                      new gunrock::gcuda::multi_context_t(0)) // Context
    )
    {
        // <user-defined>

        using vertex_t = typename graph_t::vertex_type;

        using param_type = param_t<vertex_t>;
        using result_type = result_t<vertex_t>;

        param_type param(mask);
        result_type result(marker);
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