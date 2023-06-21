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

void rasterScan(cv::Mat &marker, cv::Mat &mask)
{
    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {
            int n_plus_neighbors[4][2] = {
                {j - 1, i},
                {j - 1, i - 1},
                {j, i - 1},
                {j + 1, i - 1}};

            int p_value = (int)marker.at<uchar>(j, i);
            int m_value = (int)mask.at<uchar>(j, i);

            for (int n = 0; n < 4; n++)
            {
                if (n_plus_neighbors[n][0] < 0 ||
                    n_plus_neighbors[n][0] > marker.cols - 1 ||
                    n_plus_neighbors[n][1] < 0 ||
                    n_plus_neighbors[n][1] > marker.rows - 1) // checking out-of-bounds
                {
                    continue;
                }

                int n_value = (int)marker.at<uchar>(n_plus_neighbors[n][0], n_plus_neighbors[n][1]);
                p_value = std::max(p_value, n_value);
            }

            p_value = std::min(p_value, m_value);
            marker.at<uchar>(j, i) = p_value;
        }
    }
}

void antiRasterScan(cv::Mat &marker, cv::Mat &mask)
{
    std::vector<int> fifo;
    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {
            int n_minus_neighbors[4][2] = {
                {j - 1, i + 1},
                {j, i + 1},
                {j + 1, i + 1},
                {j + 1, i}};

            int p_value = (int)marker.at<uchar>(j, i);
            int m_value = (int)mask.at<uchar>(j, i);

            for (int n = 0; n < 4; n++)
            {
                if (n_minus_neighbors[n][0] < 0 ||
                    n_minus_neighbors[n][0] > marker.cols - 1 ||
                    n_minus_neighbors[n][1] < 0 ||
                    n_minus_neighbors[n][1] > marker.rows - 1) // checking out-of-bounds
                {
                    continue;
                }

                int n_value = (int)marker.at<uchar>(n_minus_neighbors[n][0], n_minus_neighbors[n][1]);
                p_value = std::max(p_value, n_value);
            }

            p_value = std::min(p_value, m_value);
            marker.at<uchar>(j, i) = p_value;

            for (int n = 0; n < 4; n++)
            {
                if (n_minus_neighbors[n][0] < 0 ||
                    n_minus_neighbors[n][0] > marker.cols - 1 ||
                    n_minus_neighbors[n][1] < 0 ||
                    n_minus_neighbors[n][1] > marker.rows - 1) // checking out-of-bounds
                {
                    continue;
                }

                int n_value = (int)marker.at<uchar>(n_minus_neighbors[n][0], n_minus_neighbors[n][1]);
                int m_n_value = (int)mask.at<uchar>(n_minus_neighbors[n][0], n_minus_neighbors[n][1]);

                if (n_value < p_value && n_value < m_n_value)
                {
                    fifo.push_back(get1DCoords(marker, pixel_coords(j, i)));
                }
            }
        }
    }

    debug(fifo.size());
}

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

    // std::cout << marker << std::endl;

    iwp::runMorphRec(marker, mask);

    return 0;
}