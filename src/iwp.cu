#include "iwp.hxx"

int iwp::get1DCoords(cv::Mat &img, pixel_coords coords)
{
    return (coords.second * img.size().width) + coords.first;
}

pixel_coords iwp::get2DCoords(int width, int coord)
{
    return pixel_coords(coord % width, coord / width);
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
auto iwp::convertImgToGraph(cv::Mat &marker, cv::Mat &mask, thrust::device_vector<vertex_t> &markerValues, thrust::device_vector<vertex_t> &maskValues)
{
    using csr_t = gunrock::format::csr_t<gunrock::memory_space_t::device, vertex_t, edge_t, weight_t>;

    // Allocate space for vectors
    gunrock::vector_t<edge_t, gunrock::memory_space_t::host> Ap;   // rowOffset
    gunrock::vector_t<vertex_t, gunrock::memory_space_t::host> Aj; // columnIdx
    gunrock::vector_t<weight_t, gunrock::memory_space_t::host> Ax; // values

    int number_of_rows = 100;
    int number_of_nonzeros = 684;

    Ap.resize(number_of_rows + 1);
    Aj.resize(number_of_nonzeros);
    Ax.resize(number_of_nonzeros);

    const int HAS_EDGE = 1;

    if (marker.empty())
        throw "Unable to read image";

    Ap[0] = 0;

    int apCount = 1;
    int ajCount = 0;
    int axCount = 0;

    for (int i = 0; i < marker.rows; i++)
    {
        for (int j = 0; j < marker.cols; j++)
        {

            pixel_coords pixel = pixel_coords(j, i);
            int oneDPos = get1DCoords(marker, pixel);

            markerValues[oneDPos] = (int)marker.at<uchar>(j, i);
            maskValues[oneDPos] = (int)mask.at<uchar>(j, i);

            std::vector<int> neighbours = getPixelNeighbours(marker, pixel);
            for (int neighbour : neighbours)
            {
                Aj[ajCount] = neighbour;
                Ax[axCount] = HAS_EDGE;
                ajCount++;
                axCount++;
                // columnIdx.push_back(neighbour);
                // values.push_back(HAS_EDGE);
            }
            Ap[apCount] = Ap[apCount - 1] + neighbours.size();
            apCount++;
        }
    }

    csr_t csr(marker.rows * marker.cols, marker.rows * marker.cols, number_of_nonzeros);

    csr.row_offsets = Ap;
    csr.column_indices = Aj;
    csr.nonzero_values = Ax;

    debug(csr.row_offsets[10]);
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

    float gpu_elapsed = run(G, maskValues.data().get(), marker.cols, marker.rows, markerValues.data().get());

    gunrock::print::head(markerValues, 100, "Marker");

    return G;
}

float iwp::runMorphRec(cv::Mat &marker, cv::Mat &mask)
{
    using vertex_t = int;
    using edge_t = int;
    using weight_t = int;

    int numVertices = marker.rows * marker.cols;

    thrust::device_vector<vertex_t> markerValues(numVertices);
    thrust::device_vector<vertex_t> maskValues(numVertices);

    auto markerGraph = convertImgToGraph<vertex_t, edge_t, weight_t>(marker,
                                                                     mask,
                                                                     markerValues,
                                                                     maskValues);

    // float gpu_elapsed = run(markerGraph, maskValues.data().get(), markerValues.data().get());

    // gunrock::print::head(markerValues, 20, "Marker");

    // std::cout << "GPU Elapsed: " << gpu_elapsed << std::endl;
}