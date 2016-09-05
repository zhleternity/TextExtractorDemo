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
//    waitKey();
    
    if (! imageDirectory.empty()) {
        imwrite( imageDirectory + "/out_grey.png",                   gray );
        imwrite( imageDirectory + "/out_mser_mask.png",              mserMask );
        imwrite( imageDirectory + "/out_canny_edges.png",            edges );
        imwrite( imageDirectory + "/out_edge_mser_intersection.png", edge_mser_bitand );
        imwrite( imageDirectory + "/out_gradient_grown.png",         gradGrowth );
        imwrite( imageDirectory + "/out_edge_enhanced_mser.png",     edge_enhanced_mser );
    }
    
    
    //find CC
    ConnectedComponent CC(Detectorparams.maxConnComponentNum, 8);
    cv::Mat labels = CC.apply(edge_enhanced_mser);
    imshow("labels", labels);
//    waitKey();
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
    //distance transform
    distanceTransform(result, result, CV_DIST_L2, 3);
    result.convertTo(result, CV_32SC1);
    
    //find stroke width image from the distance transform
    cv::Mat stroke_width = computeStrokeWidth(result);
    imshow("stroke width ", stroke_width);
    
    //again filter the stroke width by the CC
    ConnectedComponent conn_comp(Detectorparams.maxConnComponentNum, 8);
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
        
        //get the mean and stddev for each CC
        vector<double> tmp_mean,tmp_stddev;
        meanStdDev(nonzero, tmp_mean, tmp_stddev);
        double mean = tmp_mean[0];
        double stddev = tmp_stddev[0];
        
        if((stddev / mean) > Detectorparams.maxStdDevMeanRatio)
            continue;
        filtered_stroke_width |= mask;
        
    }
    
    cv::Mat bounidngRegion;//= filtered_stroke_width.clone();
    cv::Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, cv::Size(11,11));
    cv::Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, cv::Size(5,5));
    morphologyEx(filtered_stroke_width, bounidngRegion, MORPH_CLOSE, kernel1);
    imshow("morph close", bounidngRegion);
    
    morphologyEx(bounidngRegion, bounidngRegion, MORPH_OPEN, kernel2);
    imshow("morph open", bounidngRegion);
//    waitKey();
    
    cv::Mat boundingRectCoords;
    findNonZero(bounidngRegion, boundingRectCoords);
    cv::Rect boundingRect = cv::boundingRect(boundingRectCoords);
    cv::Mat bounding_mask(filtered_stroke_width.size(), CV_8UC1, Scalar(0));
    cv::Mat(bounding_mask, boundingRect) = 255;
    
    //add some margin to the bounding rect
    boundingRect = cv::Rect(boundingRect.tl() - cv::Point(5, 5), boundingRect.br() + cv::Point(5, 5));
    boundingRect = clamp(boundingRect, image.size());
    
    //discard everything outside of the bounding rectangle
    filtered_stroke_width.copyTo(filtered_stroke_width, bounding_mask);
    return pair<cv::Mat, cv::Rect>(filtered_stroke_width, boundingRect);
    
    
    
}


//threshold func
//clamp the bounding regioin rect in specified scope
cv::Rect TextDetector::clamp(cv::Rect &rect, cv::Size size){
    cv::Rect result = rect;
    
    if(rect.x < 0)
        result.x = 0;
    if((result.x + result.width) > size.width)
        result.width = size.width - result.x;
    if(result.y < 0)
        result.y = 0;
    if((result.y + result.height) > size.height)
        result.height = size.height - result.y;
    return result;
}


cv::Mat TextDetector::createMSERMask(cv::Mat &gray){
    //find MSER components
    vector<vector<cv::Point>> contours;
    vector<cv::Rect> boxes;
    Ptr<MSER> mser = MSER::create(8, Detectorparams.minMSERArea, Detectorparams.maxMSERArea, 0.25, 0.1, 100, 1.0, 0.03, 5);
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
                                
//segment the spine text
void TextDetector::segmentText(cv::Mat &spineImage, cv::Mat &segSpine, bool removeNoise){

    cv::Mat spineGray;
    cvtColor(spineImage, spineGray, CV_RGB2GRAY);
    cv::Mat spineAhe;
    adaptiveHistEqual(spineGray, spineAhe, 2.5);
//    Size spine_gray_sz = spineGray.size();
    
    cv::Mat spine_th(spineGray.size(), CV_8UC1, Scalar(0));
    int window_num = 40;
    int window_h = spineImage.rows / window_num;
    int window_w = spineImage.cols;
    for (int i = 0; i < window_num; i ++) {
        int cut_from_r = window_h * i;
        int cut_to_r = window_h * (i+1);
        cv::Mat window_img;
        cv::Rect rect = cv::Rect(0, cut_from_r, window_w, cut_to_r - cut_from_r);
        getROI(spineImage, window_img, rect);
        cv::Mat window_img_gray;
        cvtColor(window_img, window_img_gray, CV_RGB2GRAY);
        Laplacian(window_img_gray, window_img_gray, window_img_gray.depth());
        double max_local,min_local;
        minMaxLoc(window_img_gray, &min_local, &max_local);
        double color_diff = max_local - min_local;
        double thresh;
        cv::Mat window_tmp;
        if (color_diff > 50)
            thresh = threshold(window_img_gray, window_tmp, 1, 255, THRESH_OTSU);
        else
            thresh = 0;
        cv::Mat seg_window;
        threshold(window_img_gray, seg_window, thresh, 255, THRESH_BINARY);
        uchar *first = seg_window.ptr<uchar>(0);
        uchar *last = seg_window.ptr<uchar>(seg_window.rows - 1);
        vector<int> cols1,cols2;
        findKEdge(first, 0, 5, cols1);
        findKEdge(last , 0, 5, cols2);
        float max_zero_dist, max_one_dist;
        if(cols1.empty() || cols2.empty())
            max_zero_dist = 0.0;
        else{
            float avg_right = sum(cols2)[0] / (int)sizeof(cols2);
            float avg_left  = sum(cols1)[0] / (int)sizeof(cols1);
            max_zero_dist = avg_right - avg_left;
        }
        cols1.clear();
        cols2.clear();
        
        findKEdge(first, 1, 5, cols1);
        findKEdge(last , 1, 5, cols2);
        if(cols1.empty() || cols2.empty())
            max_one_dist = 0;
        else{
            float avg_right = sum(cols2)[0] / (int)sizeof(cols2);
            float avg_left  = sum(cols1)[0] / (int)sizeof(cols1);
            max_one_dist = avg_right - avg_left;
        }
        cv::Mat idx;
        findNonZero(seg_window, idx);
        int one_count = (int)idx.total();
        int zero_count = (int)seg_window.total() - one_count;
        
        float one_zero_diff = max_one_dist - max_zero_dist;
        float  dist_limit = 5;
        
        if(one_zero_diff > dist_limit)
            seg_window = ~ seg_window;
        else{
            if(one_zero_diff > -dist_limit && one_count > zero_count)
                seg_window = ~ seg_window;
        }
        
//        cv::Mat spine_append(spineImage.size(), CV_8UC1, Scalar(0));
        seg_window.copyTo(cv::Mat( spine_th, rect));
            
        
    }
    
    if (removeNoise) {
        vector<vector<cv::Point>> contours;
        findContours(spine_th, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        
        ConnectedComponent CC(Detectorparams.maxConnComponentNum, 8);
        cv::Mat labels = CC.apply(spine_th);
        vector<ComponentProperty> props = CC.getComponentsProperties();
        
        
        for (ComponentProperty &prop : props) {
            int box_width  = prop.boxCC.width;
            int box_height = prop.boxCC.height;
            float box_aspect = box_width / box_height;
            int box_area = prop.area;
            
            vector<cv::Point> tmp;
            tmp = prop.pixelIdxList;
            
            for (int i = 0; i < prop.pixelIdxList.size(); i ++) {
                if (box_width > spineImage.cols / 1.001 || (box_width > spineImage.cols / 1.4 && box_aspect > 5)
                    || (box_height > spineImage.cols / 1.1) || ((box_area < (spineImage.cols/30))^2)
                    || (box_aspect > 0.5 && box_aspect < 1.7 && (prop.solidity > 0.9))) {
                    spine_th.at<int>(tmp[i].x, tmp[i].y)= 0;
                    tmp.clear();
                }
            }
        }
    }
    segSpine = spine_th;
    spine_th.release();
    
    
    
}

void TextDetector::findKEdge(uchar *data, int edgeValue,int k,vector<int> &coords){
    int count = 0;
    for (int i = 0; i < (int)sizeof(data); i ++) {
        if(edgeValue == data[i]){
            if(count != k){
                count ++;
                coords.push_back(i);
            }
            else if (count == k)
                break;
        }
    }
    
}

void TextDetector::adaptiveHistEqual(cv::Mat &src,cv::Mat &dst,double clipLimit)
{
    Ptr<cv::CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->apply(src, dst);
}

void TextDetector::perspective(Mat &src, float in_point[8], Mat &dst)
{
    float w_a4 = sqrt(pow(in_point[0] - in_point[2], 2) + pow(in_point[1] - in_point[3] ,2 ));
    float h_a4 = sqrt(pow(in_point[0] - in_point[4], 2) + pow(in_point[1] - in_point[5] ,2));
    dst = Mat::zeros(h_a4, w_a4, CV_8UC3);
    
    
    
    // corners of destination image with the sequence [tl, tr, bl, br]
    vector<Point2f> dst_pts, img_pts;
    dst_pts.push_back(Point(0, 0));
    dst_pts.push_back(Point(w_a4 - 1, 0));
    dst_pts.push_back(Point(0, h_a4 - 1));
    dst_pts.push_back(Point(w_a4 - 1, h_a4 - 1));
    
    // corners of source image with the sequence [tl, tr, bl, br]
    img_pts.push_back(Point(in_point[0], in_point[1]));
    img_pts.push_back(Point(in_point[2],in_point[3]));
    img_pts.push_back(Point(in_point[4],in_point[5]));
    img_pts.push_back(Point(in_point[6], in_point[7]));
    
    
    // get transformation matrix
    Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);
    
    // apply perspective transformation
    warpPerspective(src, dst, transmtx, dst.size());
}


void TextDetector::getROI(cv::Mat &src,cv::Mat &out,cv::Rect rect)
{
    out = cv::Mat(rect.width, rect.height, CV_8UC3,Scalar(125));
    vector<cv::Point2f>  quad_pts;
    //映射到原图上
    vector<cv::Point2f> pointss;
    pointss.push_back(rect.tl());
    pointss.push_back(rect.tl() + cv::Point(rect.width,0));
    pointss.push_back(rect.tl() + cv::Point(0,rect.height));
    pointss.push_back(rect.br());
    
    //以原点为顶点的矩形
    quad_pts.push_back(cv::Point2f(0,0));
    quad_pts.push_back(cv::Point2f(rect.height,0));
    quad_pts.push_back(cv::Point2f(0,rect.width));
    quad_pts.push_back(cv::Point2f(rect.height,
                                   rect.width));
    
    //    获取透视变换的变换矩阵
    cv::Mat transmtx = getPerspectiveTransform(pointss, quad_pts);
    
    warpPerspective(src, out, transmtx,out.size());
    
}
























































