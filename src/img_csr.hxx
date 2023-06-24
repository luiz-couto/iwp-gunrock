#pragma once
#ifndef IMG_CSR
#define IMG_CSR

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

int getNumberOfEdges2(int width, int height, CONN conn)
{
    if (conn == CONN_4)
    {
        return (width * height * 4) - ((4 * 2) + ((width - 2) * 1 * 2) + ((height - 2) * 1 * 2));
    }
    return (width * height * 8) - ((4 * 5) + ((width - 2) * 3 * 2) + ((height - 2) * 3 * 2));
}

class ImageCSR
{
public:
    int img_width;
    int img_height;
    int num_edges;
    CONN conn;
    thrust::device_vector<int> row_offsets;
    thrust::device_vector<int> column_idxs;
    thrust::device_vector<int> values;

    ImageCSR(int _img_width, int _img_height, CONN _conn)
    {

        this->img_width = _img_width;
        this->img_height = _img_height;
        this->conn = _conn;
        this->num_edges = getNumberOfEdges2(this->img_width, this->img_height, this->conn);

        thrust::device_vector<int> _row_offsets(this->img_width * this->img_height + 1, 0);
        this->row_offsets = _row_offsets;
        setRowOffsets(this->row_offsets.data().get(), this->img_width, this->img_height, this->conn);

        int max_num_ngbs = 8;
        if (conn == CONN_4)
        {
            max_num_ngbs = 4;
        }
        thrust::device_vector<int> _column_idxs(this->img_width * this->img_height * max_num_ngbs, -1);
        this->column_idxs = _column_idxs;
        setColumnIdxs(this->column_idxs.data().get(), this->img_width, this->img_height, this->conn);

        cudaDeviceSynchronize();

        thrust::remove(this->column_idxs.begin(), this->column_idxs.end(), -1);
        this->column_idxs.resize(this->num_edges);

        thrust::device_vector<int> _values(this->num_edges, 1);
        this->values = _values;
    }

    void setRowOffsets(int *row_offsets_ptr, int img_width, int img_height, CONN conn)
    {
        auto set_row_offset = [img_width, img_height, conn, row_offsets_ptr] __device__(int const &v)
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

        thrust::for_each(thrust::device, thrust::make_counting_iterator<int>(0),      // Begin: 0
                         thrust::make_counting_iterator<int>(img_width * img_height), // End: # of Vertices
                         set_row_offset);                                             // Unary operation
    }

    void setColumnIdxs(int *column_idxs_ptr, int img_width, int img_height, CONN conn)
    {
        int max_num_ngbs = 8;
        if (conn == CONN_4)
        {
            max_num_ngbs = 4;
        }

        auto set_column_idx = [img_width, img_height, conn, max_num_ngbs, column_idxs_ptr] __device__(int const &v)
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

        thrust::for_each(thrust::device, thrust::make_counting_iterator<int>(0),      // Begin: 0
                         thrust::make_counting_iterator<int>(img_width * img_height), // End: # of Vertices
                         set_column_idx);                                             // Unary operation
    }
};

#endif