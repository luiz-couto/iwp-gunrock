#pragma once
#include <opencv2/core.hpp>
#include <gunrock/algorithms/algorithms.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;
using graph_t = gunrock::graph::graph_csr_t<vertex_t, edge_t, weight_t>;
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
    std::vector<int> getPixelNeighbours(cv::Mat &img, pixel_coords coords);
    graph_t convertImgToGraph(cv::Mat &img);
    float runMorphRec(cv::Mat &marker, cv::Mat &mask);

    struct param_t
    {
        cv::Mat &mask;
        param_t(cv::Mat &_mask) : mask(_mask) {}
    };

    struct result_t
    {
        cv::Mat &result_img;
        result_t(cv::Mat _result_img) : result_img(_result_img) {}
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

        using vertex_t = typename graph_t::vertex_type;
        using edge_t = typename graph_t::edge_type;
        using weight_t = typename graph_t::weight_type;

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
            auto P = this->get_problem();
            f->push_back(P->param.single_source);
        }

        void loop(gunrock::gcuda::multi_context_t &context) override
        {
        }

    }; // struct enactor_t
}