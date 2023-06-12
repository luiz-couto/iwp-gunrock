#pragma once
#include <iostream>
#include <opencv2/core.hpp>

namespace iwp
{
    namespace examples
    {

        cv::Mat genSameValueImg(int width, int height, int value)
        {
            return cv::Mat(width, height, CV_8UC1, value);
        }

        /*
        Gens the 10 x 10 image below
        08 08 08 08 08 08 08 08 08 08
        08 12 12 12 08 08 09 08 09 08
        08 12 12 12 08 08 08 09 08 08
        08 12 12 12 08 08 09 08 09 08
        08 08 08 08 08 08 08 08 08 08
        08 09 08 08 08 16 16 16 08 08
        08 08 08 09 08 16 16 16 08 08
        08 08 09 08 08 16 16 16 08 08
        08 09 08 09 08 08 08 08 08 08
        08 08 08 08 08 08 09 08 08 08
        */
        cv::Mat genBigMarkerImg()
        {
            cv::Mat marker = cv::Mat(10, 10, CV_8UC1, 8);

            for (int i = 1; i < 4; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    marker.at<uchar>(i, j) = 12;
                }
            }

            for (int i = 5; i < 8; i++)
            {
                for (int j = 5; j < 8; j++)
                {
                    marker.at<uchar>(i, j) = 16;
                }
            }

            marker.at<uchar>(5, 1) = 9;
            marker.at<uchar>(8, 1) = 9;
            marker.at<uchar>(7, 2) = 9;
            marker.at<uchar>(6, 3) = 9;
            marker.at<uchar>(8, 3) = 9;
            marker.at<uchar>(1, 6) = 9;
            marker.at<uchar>(3, 6) = 9;
            marker.at<uchar>(9, 6) = 9;
            marker.at<uchar>(2, 7) = 9;
            marker.at<uchar>(1, 8) = 9;
            marker.at<uchar>(3, 8) = 9;

            return marker;
        }

        /*
        Gens the 10 x 10 image below
        10 10 10 10 10 10 10 10 10 10
        10 14 14 14 10 10 11 10 11 10
        10 14 14 14 10 10 10 11 10 10
        10 14 14 14 10 10 11 10 11 10
        10 10 10 10 10 10 10 10 10 10
        10 11 10 10 10 16 16 16 10 10
        10 10 10 11 10 16 16 16 10 10
        10 10 11 10 10 16 16 16 10 10
        10 11 10 11 10 10 10 10 10 10
        10 10 10 10 10 10 11 10 10 10
        */
        cv::Mat genBigMaskImg()
        {
            cv::Mat mask = cv::Mat(10, 10, CV_8UC1, 10);

            for (int i = 1; i < 4; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    mask.at<uchar>(i, j) = 14;
                }
            }

            for (int i = 5; i < 8; i++)
            {
                for (int j = 5; j < 8; j++)
                {
                    mask.at<uchar>(i, j) = 16;
                }
            }

            mask.at<uchar>(5, 1) = 11;
            mask.at<uchar>(8, 1) = 11;
            mask.at<uchar>(7, 2) = 11;
            mask.at<uchar>(6, 3) = 11;
            mask.at<uchar>(8, 3) = 11;
            mask.at<uchar>(1, 6) = 11;
            mask.at<uchar>(3, 6) = 11;
            mask.at<uchar>(9, 6) = 11;
            mask.at<uchar>(2, 7) = 11;
            mask.at<uchar>(1, 8) = 11;
            mask.at<uchar>(3, 8) = 11;

            return mask;
        }

        /*
        Gens the 10 x 10 image below
        10 10 10 10 10 10 10 10 10 10
        10 12 12 12 10 10 10 10 10 10
        10 12 12 12 10 10 10 10 10 10
        10 12 12 12 10 10 10 10 10 10
        10 10 10 10 10 10 10 10 10 10
        10 10 10 10 10 16 16 16 10 10
        10 10 10 10 10 16 16 16 10 10
        10 10 10 10 10 16 16 16 10 10
        10 10 10 10 10 10 10 10 10 10
        10 10 10 10 10 10 10 10 10 10
        */
        cv::Mat genExpectedImg()
        {
            cv::Mat expected = cv::Mat(10, 10, CV_8UC1, 10);

            for (int i = 1; i < 4; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    expected.at<uchar>(i, j) = 12;
                }
            }

            for (int i = 5; i < 8; i++)
            {
                for (int j = 5; j < 8; j++)
                {
                    expected.at<uchar>(i, j) = 16;
                }
            }

            std::cout << expected << std::endl;
            return expected;
        }

    }
}