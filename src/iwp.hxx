#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/container/vector.hxx>
#include <thrust/for_each.h>
#include <cstdio>
#include <map>
#include <algorithm> // std::min

#define log(msg, x) std::cout << msg << ": " << #x << " = " << x << std::endl;
#define debug(x) std::cout << #x << " = " << x << std::endl;
#define debug2(x, y) std::cout << #x << " = " << x << " --- " << #y << " = " << y << "\n";
#define debugLine(i) std::cout << "PASSOU AQUIIII" \
                               << " --- " << i << std::endl;

enum CONN
{
    CONN_4,
    CONN_8
};

enum RASTER_TYPE
{
    RASTER,
    ANTI_RASTER
};

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
    std::vector<int> getPixelNeighbours(cv::Mat &img, pixel_coords coords, CONN conn);
    int getNumberOfEdges(int width, int height, CONN conn);
    void rasterScan(cv::Mat &marker, cv::Mat &mask, CONN conn);

    template <typename vertex_t>
    std::vector<vertex_t> antiRasterScan(cv::Mat &marker, cv::Mat &mask, CONN conn);

    template <typename vertex_t>
    void saveMarkerImg(thrust::device_vector<vertex_t> &markerValues, int img_width, int img_height);

    template <typename vertex_t>
    void saveDistTransformResult(thrust::device_vector<vertex_t> &vr_diagram, int img_width, int img_height);

    template <typename vertex_t, typename edge_t, typename weight_t>
    void buildGraphAndRun(cv::Mat &marker, cv::Mat &mask, CONN conn);

    float runMorphRec(cv::Mat &marker, cv::Mat &mask);
    void runDistTransform(cv::Mat &bin_img);

    template <typename vertex_t>
    struct param_t
    {
        vertex_t *mask;
        int img_width;
        int img_height;
        param_t(vertex_t *_mask,
                int _img_width,
                int _img_height) : mask(_mask), img_width(_img_width), img_height(_img_height) {}
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
            int width = P->param.img_width;
            int height = P->param.img_height;

            int num_parts = 256;
            int num_scans = 2;

            auto update_pixel = [G, marker, mask, width, height] __device__(vertex_t const &v, RASTER_TYPE r_type)
            {
                // printf("v: %d, ", v);
                auto startEdge = G.get_starting_edge(v);
                auto numberNgbs = G.get_number_of_neighbors(v);
                vertex_t greater = marker[v];

                for (auto e = startEdge; e < startEdge + numberNgbs; e++)
                {
                    vertex_t ngb = G.get_destination_vertex(e);

                    if (r_type == RASTER && ngb > v)
                    {
                        continue;
                    }

                    if (r_type == ANTI_RASTER && ngb < v)
                    {
                        continue;
                    }

                    if (marker[ngb] > greater)
                        greater = marker[ngb];
                }

                if (greater > mask[v])
                    greater = mask[v];

                gunrock::math::atomic::exch(&marker[v], greater);
            };

            auto part_1 = [G, marker, mask, width, height, num_parts, update_pixel] __device__(int const &c)
            {
                int part_height = height / num_parts;
                int begin = c * part_height * width;
                int end = (((c + 1) * part_height) + 1) * width;

                if (c == num_parts - 1)
                {
                    end -= 1 * width;
                }

                for (int v = begin; v < end; v++)
                {
                    update_pixel(v, RASTER);
                }
            };

            auto part_2 = [G, marker, mask, width, height, num_parts, update_pixel] __device__(int const &c)
            {
                int real_c = abs(c - (num_parts - 1));
                int part_height = height / num_parts;
                int begin = real_c * part_height * width;
                int end = (((real_c + 1) * part_height) + 1) * width;

                if (real_c == num_parts - 1)
                {
                    end -= 1 * width;
                }

                for (int v = end - 1; v >= begin; v--)
                {
                    update_pixel(v, ANTI_RASTER);
                }
            };

            thrust::device_vector<vertex_t> device_frontier(width * height, -1);
            vertex_t *f_pointer = device_frontier.data().get();

            auto part_3 = [G, marker, mask, f_pointer] __device__(int const &v)
            {
                edge_t startEdge = G.get_starting_edge(v);
                auto numberNgbs = G.get_number_of_neighbors(v);

                for (edge_t e = startEdge; e < startEdge + numberNgbs; e++)
                {
                    vertex_t ngb = G.get_destination_vertex(e);
                    if ((marker[ngb] < marker[v]) && (marker[ngb] < mask[ngb]))
                    {
                        // printf("v: %d, ngb: %d", v, ngb);
                        f_pointer[v] = v;
                    }
                }
            };

            for (int i = 0; i < num_scans; i++)
            {
                thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0), thrust::make_counting_iterator<vertex_t>(num_parts), part_1);
                cudaDeviceSynchronize();

                thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0), thrust::make_counting_iterator<vertex_t>(num_parts), part_2);
                cudaDeviceSynchronize();
            }

            thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0), thrust::make_counting_iterator<vertex_t>(G.get_number_of_vertices()), part_3);
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

            vertex_t *marker = P->result.marker;
            vertex_t *mask = P->param.mask;

            auto advance_op = [marker, mask] __host__ __device__(
                                  vertex_t const &source,   // ... source
                                  vertex_t const &neighbor, // neighbor
                                  edge_t const &edge,       // edge
                                  weight_t const &weight    // weight (tuple).
                                  ) -> bool
            {
                if (marker[neighbor] < marker[source] && mask[neighbor] != marker[neighbor])
                {
                    vertex_t old_val = gunrock::math::atomic::max(&marker[neighbor], std::min(marker[source], mask[neighbor]));
                    if (old_val < std::min(marker[source], mask[neighbor]))
                    {
                        return true;
                    }
                }
                return false;
            };

            gunrock::operators::advance::execute<gunrock::operators::load_balance_t::thread_mapped>(G, E, advance_op, context);
        }

    }; // struct enactor_t

    template <typename graph_t>
    float run(graph_t &G,
              typename graph_t::vertex_type *mask,   // Parameter
              const int img_width,                   // Parameter
              const int img_height,                  // Parameter
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

        param_type param(mask, img_width, img_height);
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