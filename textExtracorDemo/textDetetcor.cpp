//
//  textDetetcor.cpp
//  textExtracorDemo
//
//  Created by lingLong on 16/8/29.
//  Copyright © 2016年 ling. All rights reserved.
//

#include "textDetetcor.hpp"
#include "ConnectedComponent.h"

using namespace std;
using namespace cv;



TextDetector::TextDetector(TextDetecorParams &params, string image_dir){
    Detectorparams = params;
    imageDirectory = image_dir;
    
}


cv::Mat TextDetector::preProcess(cv::Mat &image){
    cv::Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    return gray;
}


pair<cv::Mat, cv::Rect> TextDetector::applyTo(cv::Mat &image){
    cv::Mat gray = preProcess(image);
    imshow("gray image", gray);
    cv::Mat mserMask = createMSERMask(gray);
    imshow("mser mask ", mserMask);
    
    cv::Mat edges;
    Canny(gray, edges, Detectorparams.cannyThresh1, Detectorparams.cannyThresh2);
    imshow("canny image", edges);
    
    cv::Mat edge_mser_bitand = edges & mserMask;
    cv::Mat gradGrowth = growEdges(gray, edge_mser_bitand);
    imshow("grad growth", gradGrowth);
    cv::Mat edge_enhanced_mser = ~ gradGrowth & mserMask;
    imshow("enhance mser", edge_enhanced_mser);
    
    if (! imageDirectory.empty()) {
        imwrite( imageDirectory + "/out_grey.png",                   gray );
        imwrite( imageDirectory + "/out_mser_mask.png",              mserMask );
        imwrite( imageDirectory + "/out_canny_edges.png",            edges );
        imwrite( imageDirectory + "/out_edge_mser_intersection.png", edge_mser_bitand );
        imwrite( imageDirectory + "/out_gradient_grown.png",         gradGrowth );
        imwrite( imageDirectory + "/out_edge_enhanced_mser.png",     edge_enhanced_mser );
    }
    
    ConnectedComponentsTypes CC(Detectorparams.maxConnComponentNum, 8);
    
    
    
    
}
























































