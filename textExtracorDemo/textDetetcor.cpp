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
//    waitKey();
    //use MSER to get MSER region
    cv::Mat mserMask = createMSERMask(gray);
    imshow("mser mask ", mserMask);
//    waitKey();
    //use canny to extract edges
    cv::Mat edges;
    Canny(gray, edges, Detectorparams.cannyThresh1, Detectorparams.cannyThresh2);
    imshow("canny image", edges);
//    waitKey();
    
    //enhance the mser region using region growth
    cv::Mat edge_mser_bitand = edges & mserMask;
    cv::Mat gradGrowth = growEdges(gray, edge_mser_bitand);
    imshow("grad growth", gradGrowth);
//    waitKey();
    cv::Mat edge_enhanced_mser = ~ gradGrowth & mserMask;
    imshow("enhance mser", edge_enhanced_mser);
    waitKey();
    
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
    waitKey();
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


//convert angle to our neighbor encoding
// * | 2 | 3 | 4 |
//*  | 1 | 0 | 5 |
//*  | 8 | 7 | 6 |
int TextDetector::neighborsEncode(const float angle, const int neighbors){
     const float divisor = 180.0 / neighbors;
     return static_cast<int>(((floor(angle / divisor) - 1) / 2) + 1) % neighbors + 1;
    
}

cv::Mat TextDetector::growEdges(cv::Mat &image, cv::Mat &edge){
    CV_Assert(edge.type() == CV_8UC1);
    cv::Mat gradX, gradY;
    Sobel(image, gradX, CV_32FC1, 1, 0);
    Sobel(image, gradY, CV_32FC1, 0, 1);
    cv::Mat gradMagnitude, gradDirection;
    cartToPolar(gradX, gradY, gradMagnitude, gradDirection,true);
    /* Convert the angle into predefined 3x3 neighbor locations
     | 2 | 3 | 4 |
     | 1 | 0 | 5 |
     | 8 | 7 | 6 |
     */
    for (int i = 0; i < gradDirection.rows; i ++) {
        float *grad_ptr = gradDirection.ptr<float>(i);
        for (int j = 0; j < gradDirection.cols; j ++) {
            if(grad_ptr[j] != 0)
                grad_ptr[j] = neighborsEncode(grad_ptr[j]);
        }
    }
    gradDirection.convertTo(gradDirection, CV_8UC1);
    
    //perform region growing based on the gradient direction
    cv::Mat result = edge.clone();
    uchar *prev_ptr = result.ptr<uchar>(0);//first row
    uchar *curr_ptr = result.ptr<uchar>(1);//second row
    
    for (int i = 1; i < edge.rows - 1; i ++) {
        uchar *edge_ptr = edge.ptr<uchar>(i);
        uchar *grad_ptr1 = gradDirection.ptr<uchar>(i);
        uchar *next_ptr = result.ptr<uchar>(i + 1);//third row
        
        for (int j = 1; j < edge.cols - 1; j ++) {
            //only consider the contours
            if(edge_ptr[j] != 0)
            {
                //switch
                switch (grad_ptr1[j]) {
                    case 1:
                        curr_ptr[j-1] = 255;
                        break;
                    case 2:
                        prev_ptr[j-1] = 255;
                        break;
                    case 3:
                        prev_ptr[j] = 255;
                        break;
                    case 4:
                        prev_ptr[j+1] = 255;
                        break;
                    case 5:
                        curr_ptr[j] = 255;
                        break;
                    case 6:
                        next_ptr[j+1] = 255;
                        break;
                    case 7:
                        next_ptr[j] = 255;
                        break;
                    case 8:
                        next_ptr[j-1] = 255;
                        break;
                        
                    default:
                        break;
                }
            }
        }
        prev_ptr = curr_ptr;
        curr_ptr = next_ptr;
    }
    return result;
    
}


//convert encoded 8 bit uchar encoding to the 8 neighbors coordinates
vector<cv::Point> TextDetector::convertToCoordinates(int x, int y, bitset<8> neighbors){
    vector<cv::Point> coords;
    //the current point is (x, y)
    if( neighbors[0] )
        coords.push_back( Point(x - 1, y    ) );
    if( neighbors[1] )
        coords.push_back( Point(x - 1, y - 1) );
    if( neighbors[2] )
        coords.push_back( Point(x    , y - 1) );
    if( neighbors[3] )
        coords.push_back( Point(x + 1, y - 1) );
    if( neighbors[4] )
        coords.push_back( Point(x + 1, y    ) );
    if( neighbors[5] )
        coords.push_back( Point(x + 1, y + 1) );
    if( neighbors[6] )
        coords.push_back( Point(x    , y + 1) );
    if( neighbors[7] )
        coords.push_back( Point(x - 1, y + 1) );
    return coords;
}

vector<cv::Point> TextDetector::convertToCoordinates(cv::Point &point, bitset<8> neighbors){
    return convertToCoordinates(point.x, point.y, neighbors);
}
vector<cv::Point> TextDetector::convertToCoordinates(cv::Point &point, uchar neighbors){
    return convertToCoordinates(point.x, point.y, bitset<8>(neighbors));
}

//get a set of 8 neighbors that are less than current vaue
inline bitset<8> TextDetector::getMinNeighbors(int *curr_ptr, int x, int *prev_ptr, int *next_ptr){
    bitset<8> neighbor;
    neighbor[0] = curr_ptr[x-1] == 0 ? 0 : curr_ptr[x-1] < curr_ptr[x];
    neighbor[1] = prev_ptr[x-1] == 0 ? 0 : prev_ptr[x-1] < curr_ptr[x];
    neighbor[2] = prev_ptr[x  ] == 0 ? 0 : prev_ptr[x  ] < curr_ptr[x];
    neighbor[3] = prev_ptr[x+1] == 0 ? 0 : prev_ptr[x+1] < curr_ptr[x];
    neighbor[4] = curr_ptr[x+1] == 0 ? 0 : curr_ptr[x+1] < curr_ptr[x];
    neighbor[5] = next_ptr[x+1] == 0 ? 0 : next_ptr[x+1] < curr_ptr[x];
    neighbor[6] = next_ptr[x  ] == 0 ? 0 : next_ptr[x  ] < curr_ptr[x];
    neighbor[7] = next_ptr[x-1] == 0 ? 0 : next_ptr[x-1] < curr_ptr[x];
    return neighbor;
    
}

//compute stroke width image out from the distance transform matrix.
//It will propagate the max values of each connected component from the ridge to outer boundaries.
cv::Mat TextDetector::computeStrokeWidth(cv::Mat &dst){
    //pad the distance transform matrix to avoid boundary checking
    cv::Mat padded(dst.rows + 1, dst.cols + 1, dst.type(), Scalar(0));
    dst.copyTo(cv::Mat(padded, cv::Rect(1,1,dst.cols,dst.rows)));
    cv::Mat lookup(padded.size(), CV_8UC1, Scalar(0));
    int *prev_ptr = padded.ptr<int>(0);//first row
    int *curr_ptr = padded.ptr<int>(1);// second row
    
    for (int i = 1; i < padded.rows - 1; i ++){
        uchar *lookup_ptr = lookup.ptr<uchar>(i);
        int *next_ptr = padded.ptr<int>(i+1);//third row
        for (int j = 1; j < padded.cols - 1; j ++){
            //extract all the neighbors whicih value < curr_ptr[x], encoded into 8-bit uchar
            if (curr_ptr[j]){
                lookup_ptr[j] = static_cast<uchar>(getMinNeighbors(curr_ptr, j, prev_ptr, next_ptr).to_ullong());//convert bitset<8> to decimal with 8-bit uchar type
            }
        }
            prev_ptr = curr_ptr;//next loop
            curr_ptr = next_ptr;
    }
        
        
    //get max stroke width from distance transform
    double max_val_double;
    //find the local max value
    minMaxLoc(padded, 0, &max_val_double);
    int max_stroke = static_cast<int>(round(max_val_double));
    
    for (int i = max_stroke; i > 0; i --){
        cv::Mat stroke_idx_mat;
        findNonZero(padded == i, stroke_idx_mat);
        
        vector<cv::Point> stroke_idx;
        stroke_idx_mat.copyTo(stroke_idx);
        
        vector<cv::Point> neighbors;
        for (cv::Point &stroke_index : stroke_idx ){
            vector<cv::Point> tmp = convertToCoordinates(stroke_index, lookup.at<uchar>(stroke_index));
            neighbors.insert(neighbors.end(), tmp.begin(), tmp.end());
        }
        
        while (! neighbors.empty()){
            for (cv::Point &neighbor: neighbors){
                padded.at<int>(neighbor) = i;
            }
            
            neighbors.clear();
            
            vector<cv::Point> temp(neighbors);
            neighbors.clear();
            
            //recursively gets b=neighbors of the current neighbors
            for(cv::Point &pt: temp){
                vector<cv::Point> tmp = convertToCoordinates(pt, lookup.at<uchar>(pt));
                neighbors.insert(neighbors.end(), tmp.begin(), tmp.end());
            }
        }
    }
 
    return cv::Mat(padded, cv::Rect(1,1,dst.cols, dst.rows));
}
                                

























































