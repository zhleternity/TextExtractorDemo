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
    
    ConnectedComponent CC(Detectorparams.maxConnComponentNum, 8);
    cv::Mat labels = CC.apply(edge_enhanced_mser);
    imshow("labels", labels);
    vector<ComponentProperty> propertys = CC.getComponentsProperties();
    cv::Mat result(labels.size(), CV_8UC1, Scalar(0));
    for (int i = 0; i < propertys.size(); i ++) {
        if (propertys[i].area < Detectorparams.minConnComponentArea || propertys[i].area > Detectorparams.maxConnComponentArea) {
            continue;
        }
        if (propertys[i].eccentricity < Detectorparams.minEccentricity || propertys[i].eccentricity > Detectorparams.maxEccentricity) {
            continue;
        }
        if(propertys[i].solidity < Detectorparams.minSolidity)
            continue;
        result |= (labels == propertys[i].labelID);
    }
    
    distanceTransform(result, result, CV_DIST_L2, 3);
    result.convertTo(result, CV_32SC1);
    
    
    cv::Mat stroke_width = computeStrokeWidth(result);
    imshow("stroke width ", stroke_width);
    
    ConnectedComponent conn_comp(Detectorparams.maxConnComponentNum, 4);
    labels = conn_comp.apply(stroke_width);
    propertys = conn_comp.getComponentsProperties();
    
    cv::Mat filtered_stroke_width(stroke_width.size(), CV_8UC1, Scalar(0));
    for (int i = 0; i < propertys.size(); i ++) {
        cv::Mat mask = (labels == propertys[i].labelID);
        cv::Mat tmp;
        stroke_width.copyTo(tmp, mask);
//        int cnt  = countNonZero(tmp);
        vector<int> reshape = cv::Mat(tmp.reshape(1,tmp.rows * tmp.cols));
        vector<int> nonzero;
        copy_if(reshape.begin(), reshape.end(), back_inserter(nonzero), [&](int value){return value > 0;});
        
        vector<double> tmp_mean,tmp_stddev;
        
        meanStdDev(nonzero, tmp_mean, tmp_stddev);
        double mean = tmp_mean[0];
        double stddev = tmp_stddev[0];
        
        if((stddev / mean) > Detectorparams.maxStdDevMeanRatio)
            continue;
        filtered_stroke_width |= mask;
        
    }
    
    cv::Mat bounidngRegion;
    cv::Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, cv::Size(25,25));
    cv::Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, cv::Size(7,7));
    morphologyEx(filtered_stroke_width, bounidngRegion, MORPH_CLOSE, kernel1);
    imshow("morph close", bounidngRegion);
    
    morphologyEx(bounidngRegion, bounidngRegion, MORPH_OPEN, kernel2);
    imshow("morph open", bounidngRegion);
    
    cv::Mat boundingRectCoords;
    findNonZero(bounidngRegion, boundingRectCoords);
    cv::Rect boundingRect = cv::boundingRect(boundingRectCoords);
    cv::Mat bounding_mask(filtered_stroke_width.size(), CV_8UC1, Scalar(0));
    cv::Mat(bounding_mask, boundingRect) = 255;
    
    //add some margin to the bounding rect
    boundingRect = cv::Rect(boundingRect.tl() - cv::Point(5, 5), boundingRect.br() + cv::Point(5, 5));
    cv::Mat(bounding_mask, boundingRect) = 255;
    
    //discard everything outside of the bounding rectangle
    filtered_stroke_width.copyTo(filtered_stroke_width, bounding_mask);
    return pair<cv::Mat, cv::Rect>(filtered_stroke_width, boundingRect);
    
    
    
}


cv::Mat TextDetector::createMSERMask(cv::Mat &gray){
    //find MSER components
    vector<vector<cv::Point>> contours;
    vector<cv::Rect> boxes;
    Ptr<MSER> mser = MSER::create(8, Detectorparams.minMSERArea, Detectorparams.maxMSERArea, 0.25, 0.1, 100, 1.01, 0.03, 5);
    mser->detectRegions(gray, contours, boxes);
    
    //create a binary mask out of the MSER
    cv::Mat mser_mask(gray.size(), CV_8UC1, Scalar(0));
    for (int i = 0; i < contours.size(); i ++) {
        for (cv::Point& point: contours[i])
            mser_mask.at<uchar>(point) = 255;//use white color for the stable region
    }
    
    return mser_mask;
}
























































