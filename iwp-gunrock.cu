// #include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/formats/formats.hxx>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/iwp.hxx"
#include "src/examples.hxx"
#include "opencv2/imgproc/imgproc.hpp"

namespace fs = std::filesystem;

int main()
{
    // using vertex_t = int;
    // using edge_t = int;
    // using weight_t = float;

    // std::cout << "Current path is " << std::filesystem::current_path() << '\n';

    // using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    std::string marker_path = cv::samples::findFile("../../imgs/mr/marker.png");
    cv::Mat marker = cv::imread(marker_path, cv::IMREAD_COLOR);
    if (marker.empty())
    {
        std::cout << "Could not read the image: " << marker_path << std::endl;
        return 1;
    }
    cv::cvtColor(marker, marker, 6);

    std::string mask_path = cv::samples::findFile("../../imgs/mr/mask.png");
    cv::Mat mask = cv::imread(mask_path, cv::IMREAD_COLOR);
    if (mask.empty())
    {
        std::cout << "Could not read the image: " << mask_path << std::endl;
        return 1;
    }
    cv::cvtColor(mask, mask, 6);

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