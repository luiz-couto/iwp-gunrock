#pragma once
#include <opencv2/core.hpp>
#include <gunrock/formats/formats.hxx>
// #include <gunrock/graph/build.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;
using pixel_coords = std::pair<int, int>;

namespace iwp
{
    int get1DCoords(cv::Mat &img, pixel_coords coords);
    std::vector<int> getPixelNeighbours(cv::Mat &img, pixel_coords coords);
    csr_t convertImgToGraph(cv::Mat &img);
}