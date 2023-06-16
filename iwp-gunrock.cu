// #include <gunrock/algorithms/algorithms.hxx>
// #include <gunrock/formats/formats.hxx>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/iwp.hxx"
#include "src/examples.hxx"
#include "opencv2/imgproc/imgproc.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define debug(x) std::cout << #x << " = " << x << std::endl;
#define debug2(x, y) std::cout << #x << " = " << x << " --- " << #y << " = " << y << "\n";
#define debugLine(i) std::cout << "PASSOU AQUIIII" \
                               << " --- " << i << std::endl;

using pixel_coords = std::pair<int, int>;

int get1DCoords(cv::Mat &img, pixel_coords coords)
{
    return (coords.second * img.size().width) + coords.first;
}

pixel_coords get2DCoords(int width, int coord)
{
    return pixel_coords(coord % width, coord / width);
}

namespace fs = std::filesystem;

int main()
{
    // using vertex_t = int;
    // using edge_t = int;
    // using weight_t = float;

    // std::cout << "Current path is " << std::filesystem::current_path() << '\n';

    // using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    std::string marker_path = cv::samples::findFile("../../imgs/mr/marker.png");
    cv::Mat marker = cv::imread(marker_path, cv::IMREAD_GRAYSCALE);
    if (marker.empty())
    {
        std::cout << "Could not read the image: " << marker_path << std::endl;
        return 1;
    }

    std::string mask_path = cv::samples::findFile("../../imgs/mr/mask.png");
    cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        std::cout << "Could not read the image: " << mask_path << std::endl;
        return 1;
    }

    // cv::Mat marker = iwp::examples::genBigMarkerImg();
    // cv::Mat mask = iwp::examples::genBigMaskImg();

    // debug(mask);

    // uchar *test = mask.ptr();

    // thrust::device_vector<uchar> test2(test, test + 2);

    // debug((int)test[27]);

    // debug((int)mask.at<uchar>())

    // int coord = iwp::get1DCoords(img, pixel_coords(1, 3));
    // std::cout << coord << std::endl;

    // graph_t G = iwp::convertImgToGraph(img);

    // std::cout << G.get_number_of_vertices() << std::endl;

    // cv::Mat marker = iwp::examples::genBigMarkerImg();
    // cv::Mat mask = iwp::examples::genBigMaskImg();
    // graph_t G = iwp::convertImgToGraph(test);

    // std::cout << G.get_number_of_edges() << std::endl;

    // std::cout << marker << std::endl;

    iwp::runMorphRec(marker, mask);

    return 0;
}