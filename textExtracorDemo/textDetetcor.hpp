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
#include <tesseract/basedir.h>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <<#header#>>


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
    int minConnComponentArea = 5;
    //max area of CC
    int maxConnComponentArea = 600;
    
    float minEccentricity = 0.1;
    float maxEccentricity = 0.995;
    float minSolidity = 0.4;
    float maxStdDevMeanRatio = 0.7;
    
    
};



class TextDetector{
public:
    TextDetector(string imgDir = "");
    TextDetector(TextDetecorParams &params, string imgDir = "");
    
    pair<cv::Mat, cv::Rect> applyTo(cv::Mat &image);
protected:
    //pre-processing
    cv::Mat preProcess(cv::Mat &image);
    //compute the stroke width
    cv::Mat computeStrokeWidth(cv::Mat &dst);
    //create MSER mask
    cv::Mat createMSERMask(cv::Mat &gray);
    
    static int neighborsEncode(const float angle, const int neighbours = 8);
    cv::Mat growEdges(cv::Mat &image, cv::Mat &edge);
    
    vector<cv::Point> convertToCoordinates(int x, int y , bitset<8> neighbors);
    vector<cv::Point> convertToCoordinates(cv::Point &point, bitset<8> neighbors);
    vector<cv::Point> convertToCoordinates(cv::Point &point, uchar neighbors);
    
    bitset<8> getMinNeighbors(int *curr_ptr, int x, int *prev_ptr, int *next_ptr);
    
    cv::Rect clamp(cv::Rect &rect, cv::Size size);
    void convertUtf8ToGBK(char **result, char *strUtf8);
    
private:
    string imageDirectory;
    TextDetecorParams Detectorparams;
};  /*  class TextDetector */







#endif /* textDetetcor_hpp */
