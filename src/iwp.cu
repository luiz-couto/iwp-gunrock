#include "iwp.hxx"

int iwp::get1DCoords(cv::Mat &img, pixel_coords coords)
{
    return (coords.second * img.size().width) + coords.first;
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
                neighbours.push_back(get1DCoords(img, pixel_coords(i, j))); // fix this
            }
        }
    }

    return neighbours;
}

csr_t iwp::convertImgToGraph(cv::Mat &img)
{
    const int HAS_EDGE = 1;

    if (img.empty())
        throw "Unable to read image";

    std::vector<int> columnIdx, rowOffset, values;
    rowOffset.push_back(0);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            pixel_coords pixel = pixel_coords(i, j);
            std::vector<int> neighbours = getPixelNeighbours(img, pixel);
            for (int neighbour : neighbours)
            {
                columnIdx.push_back(neighbour);
                values.push_back(HAS_EDGE);
            }
            rowOffset.push_back(rowOffset.back() + neighbours.size());
        }
    }

    csr_t csr(rowOffset.size(), columnIdx.size(), values.size());
    csr.row_offsets = rowOffset;
    csr.column_indices = columnIdx;
    csr.nonzero_values = values;

    // Build graph

    // auto G = gunrock::graph::build::from_csr<memory_space_t::device, guntock::graph::view_t::csr>(
    //     csr.number_of_rows,              // rows
    //     csr.number_of_columns,           // columns
    //     csr.number_of_nonzeros,          // nonzeros
    //     csr.row_offsets.data().get(),    // row_offsets
    //     csr.column_indices.data().get(), // column_indices
    //     csr.nonzero_values.data().get()  // values
    // );
}