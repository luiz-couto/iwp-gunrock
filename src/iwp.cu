#include "iwp.hxx"
#include "img_csr.hxx"
#include "mr.hxx"
#include "dist.hxx"
#include <chrono>

int iwp::get1DCoords(cv::Mat &img, pixel_coords coords)
{
    return (coords.second * img.size().width) + coords.first;
}

pixel_coords iwp::get2DCoords(int width, int coord)
{
    return pixel_coords(coord % width, coord / width);
}

int iwp::getNumberOfEdges(int width, int height, CONN conn)
{
    if (conn == CONN_4)
    {
        return (width * height * 4) - ((4 * 2) + ((width - 2) * 1 * 2) + ((height - 2) * 1 * 2));
    }
    return (width * height * 8) - ((4 * 5) + ((width - 2) * 3 * 2) + ((height - 2) * 3 * 2));
}

void iwp::rasterScan(cv::Mat &marker, cv::Mat &mask, CONN conn)
{
    int num_ngbs = 4;
    if (conn == CONN_4)
        num_ngbs = 2;

    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {
            int n_plus_neighbors[4][2] = {
                {j - 1, i},
                {j, i - 1},
                {j - 1, i - 1},
                {j + 1, i - 1}};

            int p_value = (int)marker.at<uchar>(j, i);
            int m_value = (int)mask.at<uchar>(j, i);

            for (int n = 0; n < num_ngbs; n++)
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
std::vector<vertex_t> iwp::antiRasterScan(cv::Mat &marker, cv::Mat &mask, CONN conn)
{
    std::vector<vertex_t> fifo;
    int num_ngbs = 4;
    if (conn == CONN_4)
        num_ngbs = 2;

    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {
            int n_minus_neighbors[4][2] = {
                {j + 1, i},
                {j, i + 1},
                {j + 1, i + 1},
                {j - 1, i + 1}};

            int p_value = (int)marker.at<uchar>(j, i);
            int m_value = (int)mask.at<uchar>(j, i);

            for (int n = 0; n < num_ngbs; n++)
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

            for (int n = 0; n < num_ngbs; n++)
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

    return fifo;
}

std::vector<int> iwp::getPixelNeighbours(cv::Mat &img, pixel_coords coords, CONN conn)
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

                if (conn == CONN_4 && i != coords.first && j != coords.second)
                {
                    continue;
                }
                neighbours.push_back(get1DCoords(img, pixel_coords(i, j)));
            }
        }
    }

    return neighbours;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void iwp::buildGraphAndRun(cv::Mat &marker, cv::Mat &mask, CONN conn)
{
    auto beg = std::chrono::high_resolution_clock::now();
    debugLine("buildGraphAndRun");
    int img_width = marker.size().width;
    int img_height = marker.size().height;
    int num_edges = getNumberOfEdges(marker.cols, marker.rows, conn);

    thrust::device_vector<vertex_t> row_offsets(img_width * img_height + 1, 0);
    vertex_t *row_offsets_ptr = row_offsets.data().get();

    int max_num_ngbs = 8;
    if (conn == CONN_4)
    {
        max_num_ngbs = 4;
    }
    thrust::device_vector<vertex_t> column_idxs(img_width * img_height * max_num_ngbs, -1);
    vertex_t *column_idxs_ptr = column_idxs.data().get();

    thrust::device_vector<vertex_t> values(num_edges, 1);
    vertex_t *values_ptr = values.data().get();

    uchar *marker_ptr = marker.ptr();
    thrust::device_vector<uchar> marker_vec(img_width * img_height);
    uchar *marker_vec_uchar_ptr = marker_vec.data().get();
    thrust::copy(marker_ptr, marker_ptr + (img_width * img_height), marker_vec.begin());
    thrust::device_vector<vertex_t> marker_vec_int(img_width * img_height);
    vertex_t *marker_vec_int_ptr = marker_vec_int.data().get();

    uchar *mask_ptr = mask.ptr();
    thrust::device_vector<uchar> mask_vec(img_width * img_height);
    uchar *mask_vec_uchar_ptr = mask_vec.data().get();
    thrust::copy(mask_ptr, mask_ptr + (img_width * img_height), mask_vec.begin());
    thrust::device_vector<vertex_t> mask_vec_int(img_width * img_height);
    vertex_t *mask_vec_int_ptr = mask_vec_int.data().get();

    auto convert_uchar_to_int = [marker_vec_uchar_ptr, marker_vec_int_ptr, mask_vec_uchar_ptr, mask_vec_int_ptr, img_width] __device__(vertex_t const &v)
    {
        int x = v % img_width;
        int y = v / img_width;

        int coord = (x * img_width) + y;

        marker_vec_int_ptr[v] = (int)marker_vec_uchar_ptr[coord];
        mask_vec_int_ptr[v] = (int)mask_vec_uchar_ptr[coord];
    };

    auto set_column_idx = [img_width, img_height, conn, max_num_ngbs, column_idxs_ptr] __device__(vertex_t const &v)
    {
        int x = v % img_width;
        int y = v / img_width;
        int floorX = x;
        int floorY = y;

        if (x != 0)
            floorX -= 1;

        if (y != 0)
            floorY -= 1;

        int count = 0;
        for (int i = floorX; i < x + 2; i++)
        {
            for (int j = floorY; j < y + 2; j++)
            {
                if (!(i == x && j == y) && (i < img_width) && (j < img_height))
                {
                    if (conn == CONN_4 && i != x && j != y)
                        continue;

                    column_idxs_ptr[(v * max_num_ngbs) + count] = (j * img_width) + i; // maybe switch this i and j
                    count++;
                }
            }
        }
    };

    auto set_row_offset = [img_width, img_height, conn, row_offsets_ptr] __device__(vertex_t const &v)
    {
        int x = v % img_width;
        int y = v / img_width;

        int num_edges_corner = 3;
        if (conn == CONN_4)
            num_edges_corner = 2;

        int num_edges_border = 5;
        if (conn == CONN_4)
            num_edges_border = 3;

        int num_edges_inner = 8;
        if (conn == CONN_4)
            num_edges_inner = 4;

        int edges_sum = 0;
        if (y != 0)
        {
            edges_sum += (2 * num_edges_corner) + (((y * 2) - 2) * num_edges_border);
            edges_sum += ((img_width - 2) * (y - 1) * num_edges_inner) + ((img_width - 2) * num_edges_border);
        }

        if (x != 0)
        {
            if (y == 0 || y == img_height - 1)
            {
                edges_sum += 1 * num_edges_corner;
                edges_sum += (x - 1) * num_edges_border;
            }
            else
            {
                edges_sum += 1 * num_edges_border;
                edges_sum += (x - 1) * num_edges_inner;
            }
        }

        int num_ngbs;
        if ((x == 0 || x == img_width - 1) && (y == 0 || y == img_height - 1))
        {
            num_ngbs = num_edges_corner;
        }
        else if (x == 0 || x == img_width - 1 || y == 0 || y == img_height - 1)
        {
            num_ngbs = num_edges_border;
        }
        else
        {
            num_ngbs = num_edges_inner;
        }

        row_offsets_ptr[v + 1] = edges_sum + num_ngbs;
    };

    thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0),      // Begin: 0
                     thrust::make_counting_iterator<vertex_t>(img_width * img_height), // End: # of Vertices
                     set_row_offset);                                                  // Unary operation

    thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0),      // Begin: 0
                     thrust::make_counting_iterator<vertex_t>(img_width * img_height), // End: # of Vertices
                     set_column_idx);                                                  // Unary operation

    thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0),      // Begin: 0
                     thrust::make_counting_iterator<vertex_t>(img_width * img_height), // End: # of Vertices
                     convert_uchar_to_int);

    cudaDeviceSynchronize();

    thrust::remove(column_idxs.begin(), column_idxs.end(), -1);
    column_idxs.resize(num_edges);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    debug(duration.count());

    // gunrock::print::head(row_offsets, 20, "row_offsets");
    // gunrock::print::head(column_idxs, 20, "column_idxs");

    auto G = gunrock::graph::build::from_csr<gunrock::memory::memory_space_t::device, gunrock::graph::view_t::csr>(
        img_width * img_height, // rows
        img_width * img_height, // columns
        num_edges,              // nonzeros
        row_offsets_ptr,        // row_offsets
        column_idxs_ptr,        // column_indices
        values_ptr              // values
    );

    float gpu_elapsed = mr::run(G, mask_vec_int_ptr, img_width, img_height, marker_vec_int_ptr);
    debug(gpu_elapsed);

    saveMarkerImg(marker_vec_int, img_width, img_height);
}

template <typename vertex_t>
void iwp::saveMarkerImg(thrust::device_vector<vertex_t> &markerValues, int img_width, int img_height)
{
    thrust::host_vector<vertex_t> hostMarker = markerValues;
    cv::Mat marker = cv::Mat(img_height, img_width, CV_8UC1, 8);

    for (int i = 0; i < hostMarker.size(); i++)
    {
        pixel_coords p = get2DCoords(img_width, i);
        marker.at<uchar>(p.first, p.second) = hostMarker[i];
    }

    cv::imwrite("../results/final_marker.png", marker);
}

template <typename vertex_t>
void iwp::saveDistTransformResult(thrust::device_vector<vertex_t> &vr_diagram, int img_width, int img_height)
{
    thrust::host_vector<vertex_t> hostVR = vr_diagram;
    cv::Mat result = cv::Mat(img_height, img_width, CV_8UC1, 8);

    for (int i = 0; i < hostVR.size(); i++)
    {
        pixel_coords p = get2DCoords(img_width, i);
        result.at<uchar>(p.second, p.first) = euclideanDistance(i, hostVR[i], img_width);
    }

    cv::imwrite("../results/dist_result.png", result);
}

float iwp::runMorphRec(cv::Mat &marker, cv::Mat &mask)
{
    auto beg = std::chrono::high_resolution_clock::now();

    using vertex_t = int;
    using edge_t = int;
    using weight_t = int;

    CONN conn = CONN_4;
    int img_width = marker.size().width;
    int img_height = marker.size().height;

    // ImageCSR *ic = new ImageCSR(marker.size().width, marker.size().height, conn);

    buildGraphAndRun<vertex_t, edge_t, weight_t>(marker, mask, conn);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    log("Final Time", duration.count());
}

void iwp::runDistTransform(cv::Mat &bin_img)
{
    using vertex_t = int;
    using edge_t = int;
    using weight_t = int;

    CONN conn = CONN_8;
    int img_width = bin_img.size().width;
    int img_height = bin_img.size().height;

    ImageCSR *ic = new ImageCSR(img_width, img_height, conn);

    uchar *bin_img_ptr = bin_img.ptr();
    thrust::device_vector<uchar> bin_img_vec(img_width * img_height);
    uchar *bin_img_vec_uchar_ptr = bin_img_vec.data().get();
    thrust::copy(bin_img_ptr, bin_img_ptr + (img_width * img_height), bin_img_vec.begin());
    thrust::device_vector<vertex_t> bin_img_vec_int(img_width * img_height);
    vertex_t *bin_img_vec_int_ptr = bin_img_vec_int.data().get();

    auto convert_uchar_to_int = [bin_img_vec_uchar_ptr, bin_img_vec_int_ptr, img_width] __device__(vertex_t const &v)
    {
        int x = v / img_width;
        int y = v % img_width;

        int coord = (x * img_width) + y;

        int value = (int)bin_img_vec_uchar_ptr[coord];
        if ((int)bin_img_vec_uchar_ptr[coord] != 0)
        {
            value = 1;
        }
        bin_img_vec_int_ptr[v] = value;
    };

    thrust::for_each(thrust::device, thrust::make_counting_iterator<vertex_t>(0),      // Begin: 0
                     thrust::make_counting_iterator<vertex_t>(img_width * img_height), // End: # of Vertices
                     convert_uchar_to_int);

    cudaDeviceSynchronize();

    thrust::device_vector<vertex_t> vr_diagram(img_width * img_height, INT_MAX);

    auto G = gunrock::graph::build::from_csr<gunrock::memory::memory_space_t::device, gunrock::graph::view_t::csr>(
        img_width * img_height,       // rows
        img_width * img_height,       // columns
        ic->num_edges,                // nonzeros
        ic->row_offsets.data().get(), // row_offsets
        ic->column_idxs.data().get(), // column_indices
        ic->values.data().get()       // values
    );

    float gpu_elapsed = dist::run(G, bin_img_vec_int_ptr, img_width, img_height, vr_diagram.data().get());
    debug(gpu_elapsed);

    // gunrock::print::head(vr_diagram, 20, "vr_diagram");

    saveDistTransformResult(vr_diagram, img_width, img_height);
}