//
//  main.cpp
//  textExtracorDemo
//
//  Created by lingLong on 16/8/29.
//  Copyright © 2016年 ling. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
//#include "textDetetcor.hpp"



int main(int argc, const char * argv[]) {
    // insert code here...
    cv::Mat mat(3,2,CV_8UC1);
    mat.at<int>(0,0) = 1;
    mat.at<int>(0,1) = 2;
    mat.at<int>(1,0) = 1;
    mat.at<int>(1,1) = 4;
    mat.at<int>(2,0) = 1;
    mat.at<int>(2,1) = 1;
    cv::Mat dst = (mat == 1);
    for (int i = 0; i < dst.rows; i ++) {
        for (int j = 0; j < dst.cols; j ++) {
            std::cout<<dst.at<int>(i,j)<<std::endl;
        }
    }
    std::cout << "Hello, World!\n";
    return 0;
}
