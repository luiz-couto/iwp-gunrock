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
#include <algorithm>

#define debug(x) std::cout << #x << " = " << x << std::endl;
#define debug2(x, y) std::cout << #x << " = " << x << " --- " << #y << " = " << y << "\n";
#define debugLine(i) std::cout << "PASSOU AQUIIII" \
                               << " --- " << i << std::endl;

int main()
{
    // using vertex_t = int;
    // using edge_t = int;
    // using weight_t = float;

    // std::cout << "Current path is " << std::filesystem::current_path() << '\n';

    // using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    std::string marker_path = cv::samples::findFile("../../imgs/mr/100-percent-marker.jpg");
    cv::Mat marker = cv::imread(marker_path, cv::IMREAD_GRAYSCALE);
    if (marker.empty())
    {
        std::cout << "Could not read the image: " << marker_path << std::endl;
        return 1;
    }

    std::string mask_path = cv::samples::findFile("../../imgs/mr/100-percent-mask.jpg");
    cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        std::cout << "Could not read the image: " << mask_path << std::endl;
        return 1;
    }

    // std::string bin_img_path = cv::samples::findFile("../../imgs/dist/bin_img.png");
    // cv::Mat bin_img = cv::imread(bin_img_path, cv::IMREAD_GRAYSCALE);
    // if (bin_img.empty())
    // {
    //     std::cout << "Could not read the image: " << bin_img_path << std::endl;
    //     return 1;
    // }

    // auto beg = std::chrono::high_resolution_clock::now();
    // // ImageCSR *ic = new ImageCSR(marker.size().width, marker.size().height, CONN_4);

    // cudaDeviceSynchronize();

    // auto end = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    // debug(duration.count());

    // gunrock::print::head(ic->row_offsets, 20, "row_offsets");
    // gunrock::print::head(ic->column_idxs, 20, "column_idxs");

    // rasterScan(marker, mask);
    // antiRasterScan(marker, mask);

    // cv::Rect myRect(0, 0, 100, 100);
    // cv::Mat croppedImage = marker(myRect);

    // cv::Mat marker = iwp::examples::genBigMarkerImg();
    // cv::Mat mask = iwp::examples::genBigMaskImg();

    // debug(mask);

    // uchar *mask_ptr = mask.ptr();

    // debug((int)mask.at<uchar>(8767 / mask.cols, 8767 % mask.cols));

    // thrust::host_vector<uchar> device_vec(mask.rows * mask.cols);
    // thrust::copy(mask_ptr, mask_ptr + (mask.rows * mask.cols), device_vec.begin());

    // debug((int)device_vec[8767]);

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

    // std::cout << bin_img << std::endl;

    iwp::runMorphRec(marker, mask);
    // iwp::runDistTransform(bin_img);

    return 0;
}