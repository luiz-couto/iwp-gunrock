#include "iwp.hxx"

int iwp::get1DCoords(cv::Mat &img, pixel_coords coords)
{
    return (coords.second * img.size().width) + coords.first;
}

pixel_coords iwp::get2DCoords(int width, int coord)
{
    return pixel_coords(coord % width, coord / width);
}

int iwp::getNumberOfEdges(int width, int height)
{
    return (width * height * 8) - ((4 * 5) + ((width - 2) * 3 * 2) + ((height - 2) * 3 * 2));
}

void iwp::rasterScan(cv::Mat &marker, cv::Mat &mask)
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

template <typename vertex_t>
std::vector<vertex_t> iwp::antiRasterScan(cv::Mat &marker, cv::Mat &mask)
{
    std::vector<vertex_t> fifo;
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
                    fifo.push_back(get1DCoords(marker, pixel_coords(j, i))); // wrong, need to be coord of p
                }
            }
        }
    }

    return fifo;
}

std::vector<int> iwp::getPixelNeighbours(cv::Mat &img, pixel_coords coords)
{
    std::vector<int> neighbours;

    int floorX = coords.first;
    int floorY = coords.second;
    if (coords.first != 0)
    {
        floorX -= 1;
    }
    if (coords.second != 0)
    {
        floorY -= 1;
    }

    for (int i = floorX; i < coords.first + 2; i++)
    {
        for (int j = floorY; j < coords.second + 2; j++)
        {
            if (!(i == coords.first && j == coords.second) && (i < img.size().width) && (j < img.size().height))
            {
                neighbours.push_back(get1DCoords(img, pixel_coords(i, j)));
            }
        }
    }

    return neighbours;
}

template <typename vertex_t, typename edge_t, typename weight_t>
auto iwp::convertImgToGraph(cv::Mat &marker, cv::Mat &mask, thrust::device_vector<vertex_t> &markerValues, thrust::device_vector<vertex_t> &maskValues, std::vector<vertex_t> initial)
{
    debugLine("ConvertImgToGraph");
    using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    int num_edges = getNumberOfEdges(marker.cols, marker.rows);
    debug(num_edges);

    // Allocate space for vectors
    gunrock::vector_t<edge_t, gunrock::memory_space_t::host> Ap(marker.rows * marker.cols + 1); // rowOffset
    gunrock::vector_t<vertex_t, gunrock::memory_space_t::host> Aj(num_edges);                   // columnIdx

    std::vector<vertex_t> marker_host(marker.rows * marker.cols);
    std::vector<vertex_t> mask_host(marker.rows * marker.cols);

    const int HAS_EDGE = 1;
    int apCount = 1;
    int myCount = 0;

    if (marker.empty())
        throw "Unable to read image";

    Ap[0] = 0;

    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {

            pixel_coords pixel = pixel_coords(j, i);
            int oneDPos = get1DCoords(marker, pixel);

            marker_host[oneDPos] = (int)marker.at<uchar>(j, i);
            mask_host[oneDPos] = (int)mask.at<uchar>(j, i);

            // markerValues[oneDPos] = (int)marker.at<uchar>(j, i);
            // maskValues[oneDPos] = (int)mask.at<uchar>(j, i);

            std::vector<int> neighbours = getPixelNeighbours(marker, pixel);
            for (int neighbour : neighbours)
            {
                Aj[myCount] = neighbour;
                myCount++;
                // Aj.push_back(neighbour);
                // Ax.push_back(HAS_EDGE);
            }

            Ap[apCount] = Ap[apCount - 1] + neighbours.size();
            apCount++;
        }
    }

    debug(myCount);
    gunrock::vector_t<weight_t, gunrock::memory_space_t::host> Ax(num_edges, HAS_EDGE); // values

    debugLine("After For");

    csr_t csr(marker.rows * marker.cols, marker.rows * marker.cols, Ax.size());

    csr.row_offsets = Ap;
    csr.column_indices = Aj;
    csr.nonzero_values = Ax;

    debug(csr.number_of_rows);
    debug(csr.number_of_columns);
    debug(csr.number_of_nonzeros);
    debug(csr.row_offsets.size());
    debug(csr.column_indices.size());
    debug(csr.nonzero_values.size());

    // gunrock::print::head(Ap, 20, "Ap");
    // gunrock::print::head(Aj, 684, "Aj");
    // gunrock::print::head(Ax, 20, "Ax");

    // debug(values);

    thrust::copy(marker_host.begin(), marker_host.end(), markerValues.begin());
    thrust::copy(mask_host.begin(), mask_host.end(), maskValues.begin());

    debugLine("After Img Attrib");

    // Build graph
    auto G = gunrock::graph::build::from_csr<gunrock::memory::memory_space_t::device, gunrock::graph::view_t::csr>(
        csr.number_of_rows,              // rows
        csr.number_of_columns,           // columns
        csr.number_of_nonzeros,          // nonzeros
        csr.row_offsets.data().get(),    // row_offsets
        csr.column_indices.data().get(), // column_indices
        csr.nonzero_values.data().get()  // values
    );

    gunrock::print::head(markerValues, 100, "Marker");

    float gpu_elapsed = run(G, maskValues.data().get(), marker.cols, marker.rows, initial, markerValues.data().get());

    gunrock::print::head(markerValues, 100, "Marker");
    debug(gpu_elapsed);

    saveMarkerImg(markerValues, marker.cols, marker.rows);

    return G;
}

template <typename vertex_t>
void iwp::saveMarkerImg(thrust::device_vector<vertex_t> &markerValues, int img_width, int img_height)
{
    thrust::host_vector<vertex_t> hostMarker = markerValues;
    cv::Mat marker = cv::Mat(img_width, img_height, CV_8UC1, 8);

    for (int i = 0; i < hostMarker.size(); i++)
    {
        pixel_coords p = get2DCoords(img_width, i);
        marker.at<uchar>(p.first, p.second) = hostMarker[i];
    }

    cv::imwrite("final_marker.png", marker);
}

float iwp::runMorphRec(cv::Mat &marker, cv::Mat &mask)
{
    using vertex_t = int;
    using edge_t = int;
    using weight_t = int;

    // cv::Rect myRect(0, 0, 4096, 2048);

    // cv::Mat marker_cropped = marker(myRect);
    // cv::Mat mask_cropped = mask(myRect);

    // debug(marker_cropped.rows);

    int numVertices = marker.rows * marker.cols;

    rasterScan(marker, mask);
    std::vector<vertex_t> initial = antiRasterScan<vertex_t>(marker, mask);

    // debug(initial);

    thrust::device_vector<vertex_t> markerValues(numVertices);
    thrust::device_vector<vertex_t> maskValues(numVertices);

    auto markerGraph = convertImgToGraph<vertex_t, edge_t, weight_t>(marker,
                                                                     mask,
                                                                     markerValues,
                                                                     maskValues,
                                                                     initial);

    // float gpu_elapsed = run(markerGraph, maskValues.data().get(), markerValues.data().get());

    // gunrock::print::head(markerValues, 20, "Marker");

    // std::cout << "GPU Elapsed: " << gpu_elapsed << std::endl;
}