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
}