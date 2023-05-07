// #include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/formats/formats.hxx>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/iwp.hxx"

namespace fs = std::filesystem;

int main()
{
    using vertex_t = int;
    using edge_t = int;
    using weight_t = float;

    std::cout << "Current path is " << std::filesystem::current_path() << '\n';

    using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    std::string image_path = cv::samples::findFile("karu2.jpg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    int coord = iwp::get1DCoords(img, pixel_coords(1, 3));
    std::cout << coord << std::endl;

    return 0;
}