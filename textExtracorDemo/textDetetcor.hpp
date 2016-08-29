//
//  textDetetcor.hpp
//  textExtracorDemo
//
//  Created by lingLong on 16/8/29.
//  Copyright © 2016年 ling. All rights reserved.
//

#ifndef textDetetcor_hpp
#define textDetetcor_hpp


#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


struct TextDetecorParams {
    //min MSER area
    int minMSERArea = 10;
    //max MSER area
    int maxMSERArea = 2000;
    //first thresh of canny
    int cannyThresh1 = 20;
    //second thresh of canny
    int cannyThresh2 = 100;
    
    //max number of CC
    int maxConnComponentNum = 3000;
    //min area of CC
    
};








#endif /* textDetetcor_hpp */
