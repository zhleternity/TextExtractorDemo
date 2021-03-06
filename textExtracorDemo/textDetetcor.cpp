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
    cvtColor(spineImage, spineGray, CV_BGR2GRAY);
    imshow("gray source" , spineGray);
    spineGray = spineGray - 0.5;
//    WriteData("/Users/eternity/Desktop/未命名文件夹/gray1.txt", spineGray);
//    waitKey();
    cv::Mat spineAhe;
    adaptiveHistEqual(spineGray, spineAhe, 0.01);
    imshow("ahe", spineAhe);
//    WriteData("/Users/eternity/Desktop/未命名文件夹/gray2.txt", spineAhe);
    
    int window_num = 40;
    
    double window_h = (spineImage.rows / (double)window_num + 1e-3);
//    tmp_h = tmp_h * 1e2;
//    cout<<tmp_h<<endl;
//    tmp_h = tmp_h / 1e2;
//    cout<<tmp_h<<endl;
//    double window_h = tmp_h;
//    window_h = Round(window_h, 2);
    int window_w = spineImage.cols;
    
    cv::Mat spine_th = cv::Mat::zeros(spineGray.size(), CV_8U);
    
    for (int i = 0; i < window_num; i ++) {
        double cut_from_r = window_h * i;
        double cut_to_r = window_h * (i+1);
        cv::Mat window_img = cv::Mat::zeros(Size(cut_to_r-cut_from_r + 1, window_w), CV_8U);
        cv::Rect rect = cv::Rect(0, cut_from_r, window_w-1, cut_to_r - cut_from_r + 1);
//        getROI(spineGray, window_img, rect);
        window_img = cv::Mat(spineGray, rect);
        imshow("window section", window_img);
//        waitKey();
//        WriteData("/Users/eternity/Desktop/未命名文件夹/gray3.txt", window_img);
        
        sharpenImage(window_img, window_img);
        imshow("sharpen", window_img);
//        waitKey();
//        WriteData("/Users/eternity/Desktop/未命名文件夹/gray4.txt", window_img);
        double max_local,min_local;
        minMaxLoc(window_img, &min_local, &max_local);
        double color_diff = max_local - min_local;
        double thresh;
        cv::Mat window_tmp;
        if (color_diff > 50)
            thresh = threshold(window_img, window_tmp, 1, 255, THRESH_OTSU);
        else
            thresh = 0;
//        cout<<thresh<<endl;
        cv::Mat seg_window(window_img.size(), CV_64F);
        imgQuantize(window_img, seg_window, thresh);
//        WriteData("/Users/eternity/Desktop/未命名文件夹/quantize2.txt", seg_window);
        seg_window = seg_window == 1;
//        seg_window = seg_window / 255;

        vector<int> cols1,cols2,rows1,rows2;
        findKEdgeFirst(seg_window, 0, 5, rows1, cols1);
        findKEdgeLast (seg_window, 0, 5, rows2, cols2);
        float max_zero_dist, max_one_dist;
        if(cols1.empty() || cols2.empty())
            max_zero_dist = 0.0;
        else{
            float avg_right = (rows2[0]+rows2[1]+rows2[2]+rows2[3]+rows2[4]) / (float)sizeof(rows2);
            float avg_left  = (rows1[0]+rows1[1]+rows1[2]+rows1[3]+rows1[4]) / (float)sizeof(rows1);
            max_zero_dist = avg_right - avg_left;
        }
        cols1.clear();
        cols2.clear();
        rows1.clear();
        rows2.clear();
        
        
        findKEdgeFirst(seg_window, 255, 5, rows1, cols1);
        findKEdgeLast (seg_window, 255, 5, rows2, cols2);
        if(cols1.empty() || cols2.empty())
            max_one_dist = 0;
        else{
            float avg_right = (rows2[0]+rows2[1]+rows2[2]+rows2[3]+rows2[4]) / (float)sizeof(rows2);
            float avg_left  = (rows1[0]+rows1[1]+rows1[2]+rows1[3]+rows1[4]) / (float)sizeof(rows1);
            max_one_dist = avg_right - avg_left;
        }
        cols1.clear();
        cols2.clear();
        rows1.clear();
        rows2.clear();
        
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
//        imshow("spine_th", spine_th);
//        waitKey();
        
        
    }
    
    if (removeNoise) {
        vector<vector<cv::Point>> contours;
        imshow("spine_th", spine_th);
//        WriteData("/Users/eternity/Desktop/未命名文件夹/quantize1.txt", spine_th);
//        waitKey();
        findContours(spine_th, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i ++) {
            //compute bounding rect
            cv::Rect rect = boundingRect(contours[i]);
            double bbox_aspect = rect.width / (double)rect.height;
            int bbox_area = rect.width * rect.height;
            //compute solidity
            vector<vector<Point>> hull(1);
            convexHull( contours[i], hull[0] );
            double convex_area = contourArea(hull[0]);
            /* ... I hope this is correct ... */
            double solidity = bbox_area / convex_area;
            
            for (int j = 0; j < contours[i].size(); j ++) {
                if ( (rect.width > spineImage.cols / 1.001)
                    || (rect.width > spineImage.cols / 1.4 && bbox_aspect > 5.0)
                    || (rect.height > spineImage.cols / 1.1)
                    || (bbox_area < pow(spineImage.cols/30, 2))
                    || (bbox_aspect > 0.5 && bbox_aspect < 1.7 && solidity > 0.9) )
                    
                    spine_th.at<int>(contours[i][j].x, contours[i][j].y) = 0;
//                WriteData("/Users/eternity/Desktop/未命名文件夹/quantize2.txt", spine_th);
            }
            
            
        }
        
    }
    segSpine = spine_th;
//    transpose(segSpine, segSpine);
//    flip(segSpine, segSpine, 0);
    imshow("segspine", segSpine);
//    waitKey();
    spine_th.release();
    
    
    
}

void TextDetector::imgQuantize(cv::Mat &src, cv::Mat &dst, double level){
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);
    for (int i = 0; i < src.rows; i ++) {
        uchar *data = src.ptr<uchar>(i);
        uchar *data2 = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j ++) {
            if(data[j] <= level)
                data2[j] = 1;
            else
                data2[j] = 2;
                
        }
    }
    
}


//find words
void TextDetector::findWords(cv::Mat &seg_spine, int mergeFlag, cv::Mat &w_spine, vector<WordsStatus> &words_status){
    cv::Mat spine_th = seg_spine.clone();
    cv::Mat labels;
//    spine_th.convertTo(spine_th, CV_8U);
    connectedComponents(spine_th, labels);
    labels.convertTo(labels, CV_8U);
//
    vector<vector<cv::Point>> ccs;
    vector<Point2f> centers;
    vector<vector<cv::Point>> pixelIdxList;
    findContours(spine_th, ccs, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    int sz1 = (int)ccs.size();
    for (int i = 0; i < sz1; i ++) {
        Moments blob = cv::moments(ccs[i]);
        Point2f center = getBlobCentroid(blob);
        centers.push_back(center);
        pixelIdxList.push_back(ccs[i]);
        
    }
    int sz = (int)ccs.size();
    

//    ConnectedComponent CCs(Detectorparams.maxConnComponentNum, 8);
//    cv::Mat labels = CCs.apply(spine_th);
//    cv::Mat label = cv::Mat::zeros(spine_th.size() , CV_8UC1);
//    connectedComponents(spine_th, label);
//    vector<ComponentProperty> props = CCs.getComponentsProperties();
//    int sz = (int)props.size();
    vector<Point2f> cc_centers_vec;
    cv::Mat plot_pic(spine_th.size(), CV_8UC1, Scalar::all(255));
    cv::Mat cc_centers = cv::Mat::zeros(sz, 2, CV_64FC1);
    vector<vector<cv::Point>> cc_pixels;
//    for(ComponentProperty &prop : props){
//        cc_centers_vec.push_back(prop.centroid);
//        cc_pixels.push_back(prop.pixelIdxList);
//    }
    cc_centers_vec = centers;
    cc_centers = cv::Mat(cc_centers_vec);
    cc_pixels = pixelIdxList;
    
    cv::Mat cc_px_dist = cv::Mat::zeros(Size(sz, sz), CV_64FC1);
    for (int i = 0; i < cc_px_dist.rows - 1; i ++) {
        double *data = cc_px_dist.ptr<double>(i);
        for (int j = i + 1; j < cc_px_dist.cols; j ++) {
            cv::Mat px_j_mat;
            repeat(cc_pixels[j], 2, 1, px_j_mat);
            transpose(px_j_mat, px_j_mat);
            cv::Mat px_i_mat;
            repeat(cc_pixels[i], 2, 1, px_i_mat);
            transpose(px_i_mat, px_i_mat);
            double dist;
            min_px_dist(px_i_mat, px_j_mat, dist);
            cout<<dist<<endl;
            data[j] = dist;
            px_i_mat.release();
            px_j_mat.release();
        }
    }
    WriteData("/Users/eternity/Desktop/未命名文件夹/1.txt", cc_px_dist);
    
//    cv::Mat mat(3,2,CV_32FC1);
//    mat.at<float>(0,0) = 1000;
//    cout<<mat.at<float>(0,0)<<endl;
//    mat.at<float>(0,1) = 2;
//    mat.at<float>(1,0) = 1;
//    mat.at<float>(1,1) = 4;
//    mat.at<float>(2,0) = 1;
//    mat.at<float>(2,1) = 1;
//    WriteData("/Users/eternity/Desktop/未命名文件夹/test1.txt", mat);
//    cv::Mat trans(2,3,mat.type());
//    transpose(mat, trans);
//    WriteData("/Users/eternity/Desktop/未命名文件夹/test2.txt", trans);

    
    cv::Mat transpose;
    cv::transpose(cc_px_dist, transpose);
    WriteData("/Users/eternity/Desktop/未命名文件夹/2.txt", transpose);
    cc_px_dist = cc_px_dist + transpose;
    WriteData("/Users/eternity/Desktop/未命名文件夹/3.txt", cc_px_dist);
    
    
    for (int i = 0; i < cc_px_dist.rows; i ++) {
        cc_px_dist.at<double>(i, i) = NaN;
//        cout<<cc_px_dist.at<double>(i, i)<<endl;
    }
   
    WriteData("/Users/eternity/Desktop/未命名文件夹/cc1.txt", cc_px_dist);
    cv::Mat temp = cc_px_dist;
//    cc_poly_dist = temp;
    
    int curr_cc = 0;
    vector<int> cc_path;
    int k = 0;
    while (k < sz -1) {
        double *data = cc_px_dist.ptr<double>(curr_cc);
//        vector<int> data_row;
//        for (int i = 0; i < sizeof(data); i ++) {
//            if(i != curr_cc)
//                data_row.push_back((int)data[i]);
//        }
        double min_value = min_array(data);
        int next_cc = 0;
        for (int i = 0; i < sizeof(data); i ++) {
//            cout<<data[i]<<endl;
            if(min_value == data[i]){
                next_cc = i;
                break;
            }
        }
        
        for (int i = 0; i < sz; i ++) {
            cc_px_dist.at<double>(curr_cc, i) = NaN;
            cc_px_dist.at<double>(i, curr_cc) = NaN;
            
        }
        WriteData("/Users/eternity/Desktop/未命名文件夹/4.txt", cc_px_dist);
        
        Point2f pt1,pt2;
        pt1 = cv::Point2f(cc_centers_vec[curr_cc].x, cc_centers_vec[curr_cc].y);
        pt2 = cv::Point2f(cc_centers_vec[next_cc].x, cc_centers_vec[next_cc].y);
        circle(plot_pic, pt1, 5, Scalar(0,0,0));
        circle(plot_pic, pt2, 5, Scalar(0,0,0));
        line(plot_pic, pt1, pt2, Scalar(0,0,255),8);
        imshow("plot line", plot_pic);
//        waitKey();
        cc_path.push_back(curr_cc);
        curr_cc = next_cc;
        k = k + 1;
        
    }
    cc_path[k] = curr_cc;
    cc_px_dist = temp;
    
    
    //split into words
    float word_end_ratio = 1.6;
    int word_start = 1;
    
    vector<int> new_word;
    vector<vector<int>> words3;
    for (int l = 1; l < sz - 1; l ++) {
        int curr_cc = cc_path[l];
        int prev_cc = cc_path[l-1];
        int next_cc = cc_path[l+1];
        int dist_to_prev = (int)cc_px_dist.at<double>(curr_cc, prev_cc);
        int dist_to_next = (int)cc_px_dist.at<double>(curr_cc, next_cc);
        if (dist_to_prev > (dist_to_next * word_end_ratio) ) {
            if (l - 1 >= word_start) {
                
                for (int num = word_start; num < l; num ++) {
                    new_word.push_back(cc_path[num]);
                }
                
                word_start = l;
                words3.push_back(new_word);
                new_word.clear();
                Point2f curr_center = cc_centers_vec[curr_cc];
                Point2f prev_center = cc_centers_vec[prev_cc];
                Point2f middle_point;
                middle_point = cv::Point2f((curr_center.x + prev_center.x)/2, (curr_center.y + prev_center.y)/2);
            }
        }
        
        else if (dist_to_next > (dist_to_prev * word_end_ratio)){
            //            vector<int> new_word;
            for(int num = word_start; num < l+1; num ++){
                new_word.push_back(cc_path[num]);
            }
            word_start = l+1;
            words3.push_back(new_word);
            new_word.clear();
            Point2f curr_center = cc_centers_vec[curr_cc];
            Point2f next_center = cc_centers_vec[next_cc];
            Point2f middle_point;
            middle_point = cv::Point2f((curr_center.x + next_center.x)/2, (curr_center.y + next_center.y)/2);
        }
        
    }
    for(int num = word_start; num < sz; num ++){
        new_word.push_back(cc_path[num]);
    }
    words3.push_back(new_word);
    new_word.clear();
    for (int i = 0; i < words3.size(); i ++) {
        WordsStatus tmp;
        tmp.words = words3[i];
        tmp.length = (int)words3[i].size();
        words_status.push_back(tmp);
    }
    
    //    words_status.words = words3;
    //    words_status.length = {};
    //    words_status.dist_array = {};
    
    if (0 == mergeFlag) {
        for (int i = 0; i < words3.size(); i ++) {
//            vector<int> curr_word  = words3[i];
            WordsStatus tmp;
            tmp.words  = words3[i];
            tmp.length = (int)words3[i].size();
            words_status.push_back(tmp);
            for (int j = 0; j < words3[i].size(); j ++) {
                int curr_sym = words3[i][j];
//                cout<<curr_sym<<endl;
//                cout<<cc_pixels[curr_sym].size()<<endl;
                for (int k = 0; k < cc_pixels[curr_sym].size(); k ++){
                    cout<<cc_pixels[curr_sym][k].x<<","<<cc_pixels[curr_sym][k].y<<endl;
//                    if (<#condition#>) {
//                        <#statements#>
//                    }
                    labels.at<int>((int)cc_pixels[curr_sym][k].x, (int)cc_pixels[curr_sym][k].y) = i;
                }
            }
            
        }
//        cv::transpose(labels, labels);
//        flip(labels, labels, 0);
        imshow("labels", labels);
        w_spine = labels;
//        waitKey();
//        w_spine = cv::Mat::zeros(labels.size(), CV_8U);
//        labels *= 1.0/255;
//        cvtColor(labels, w_spine, CV_GRAY2BGR);
//        imshow("result words", w_spine);
//        cvtColor(w_spine, w_spine, CV_BGR2HSV);
//        imshow("hsv show", w_spine);
//        waitKey();
        
    }
    else if (1 == mergeFlag){//merge words
        cv::Mat cc_dist = temp;
        cv::Mat centers;
        cv::transpose(cc_centers, centers);
        cv::Mat cc_angles;
        cc_angle_mat(centers, cc_angles);
        
        for (int i = 0; i < words_status.size(); i ++)
            getWordsStatus(words_status[i].words, cc_dist, cc_angles, words_status[i]);
        
        vector<WordsStatus> merge_word_stat;
        cv::Mat merge_dist_mat, merge_angle_mat;
        mergeWords(words_status, cc_dist, cc_angles, merge_word_stat, merge_dist_mat, merge_angle_mat);
        vector<vector<int>> words;
        for (int i = 0; i < merge_word_stat.size(); i ++) {
            words.push_back(merge_word_stat[i].words);
        }
        
        for (int k = 0; k < words.size(); k ++) {
            vector<int> curr_word = words[k];
            for (int l = 0; curr_word.size(); l ++) {
                int curr_sym = curr_word[l];
                for (int num = 0; num < cc_pixels[curr_sym].size(); num ++){
//                    int idx;
//                    if (k <= 10) {
//                        idx = 200 + k;
//                    }
//                    else if (k > 10 && k <= 50 )
//                        idx = 100 + k;
//                    else if (k > 50 && k <= 100)
//                        idx = 20 + k;
//                    else
//                        idx = k - 50;
                    
                    labels.at<int>(cc_pixels[curr_sym][num].x, cc_pixels[curr_sym][num].y) = k;
                }
                
                
            }
        }
        
        
//        cv::transpose(labels, labels);
//        flip(labels, labels, 0);
        imshow("labels", labels);
        w_spine = labels;
//        labels *= 1.0/255;
//        cvtColor(labels, w_spine, CV_GRAY2BGR);
//        imshow("result words", w_spine);
//        cvtColor(w_spine, w_spine, CV_BGR2Lab);
//        imshow("hsv show", w_spine);
//        waitKey();

        words_status = merge_word_stat;
        
        words.clear();
        
    }
}


//recognize the text
string TextDetector::recognizeText(cv::Mat &w_spine, vector<WordsStatus> &words_stats){
    char *out = new char[200];
    
    vector<vector<cv::Point>> contours;
    findContours(w_spine, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    vector<Point2f> centers;
    vector<vector<cv::Point>> pixelIdx;
    vector<cv::Rect> bboxes;
    for (int i = 0; i < contours.size(); i ++) {
        Moments moment = cv::moments(contours[i]);
        centers.push_back(getBlobCentroid(moment));
        pixelIdx.push_back(contours[i]);
        bboxes.push_back(boundingRect(contours[i]));
    }
    
    vector<WordsStatus> words;
    int k = 0;
    for (int i = 0; i < words_stats.size(); i ++) {
        if(words_stats[i].length > 1){
            words.push_back(words_stats[i]);
            k = k + 1;
        }
    }
    
    //specify words orientation (horizon or vertical),and flip words if needed.
    for (int i = 0; i < words.size(); i ++) {
        vector<int> curr_word;
        curr_word = words[i].words;
        int first_char = curr_word[0];
        int last_char = curr_word[curr_word.size() - 1];
        Point2f p1 = centers[first_char];
        Point2f p2 = centers[last_char];
        float xdiff = abs(p1.x - p2.x);
        float ydiff = abs(p1.y - p2.y);
        if (xdiff > ydiff) {
            words[i].orientation = 1;//horizon orientation
            if(p1.x > p2.x){
                vector<int> flip_words;
                flip(words[i].words, flip_words, 1);//flip-lr,horizon orientation
                words[i].words = flip_words;
                flip_words.clear();
            }
                
        }
        else{
            words[i].orientation = 0;//vertical orientation
            if (p1.y > p2.y) {
                vector<int> flip_words;
                flip(words[i].words, flip_words, 1);
                words[i].words = flip_words;
                flip_words.clear();
            }
        }
    }
    
    //assign difderent colors to different words and display the result
    cv::Mat label;
    connectedComponents(w_spine, label);
    cv::Mat words_label = cv::Mat::zeros(label.size(), CV_8U);
    int pad_size = 30;
    cv::Mat words_pad;
    padImage(w_spine, words_pad, Size(pad_size, pad_size), 2);
    for (int i = 0; i < words.size(); i ++) {
        vector<int> curr_word = words[i].words;
        for (int j = 0; j < curr_word.size(); j ++) {
            int curr_char = curr_word[j];
            for (int num = 0; num < pixelIdx[curr_char].size(); num ++)
                words_label.at<int>(pixelIdx[curr_char][num].x, pixelIdx[curr_char][num].y) = 1;
        }
        vector<vector<cv::Point>> cc;
        findContours(words_label, cc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<cv::Rect> bboxes;
        for (int k = 0; k < cc.size(); k ++)
            bboxes.push_back(boundingRect(cc[k]));
        
        cv::Mat words_img;
        cvtColor(words_label, words_img, CV_GRAY2BGR);
        cv::Mat word_img;
        for (int num0 = 0; num0 < bboxes.size(); num0 ++) {
            Point2f mid = cv::Point2f(bboxes[num0].tl().x, bboxes[num0].tl().y) + cv::Point2f(0.5*bboxes[num0].width, 0.5*bboxes[num0].height);
            word_img = cv::Mat(words_img, bboxes[num0]);
            cv::Mat word_img_pad;
            padImage(word_img, word_img_pad, Size(5, 5), 2);
            if(0 == words[i].orientation)
                ImageRotate(word_img_pad, mid, 90, 0);
            
            string ocr_result = ocr(word_img_pad);
            const char *ch = ocr_result.c_str();
            out = strcat(out, ch);
        }
        
    }
    
    string string_result = string(out);
//    cout<<string_result<<endl;
    return string_result;
}

//call tesseract
string TextDetector::ocr(cv::Mat &stroke_mat){
    
    tesseract::TessBaseAPI tessearct_api;
    const char  *languagePath = "/usr/local/Cellar/tesseract/3.04.01_2/share/tessdata";
    const char *languageType = "chi";
    int nRet = tessearct_api.Init(languagePath, languageType,tesseract::OEM_DEFAULT);
    if (nRet != 0) {
        printf("初始化字库失败！");
        return string("Please re-initialize the words library");
    }
    tessearct_api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tessearct_api.SetImage(stroke_mat.data, stroke_mat.cols, stroke_mat.rows, 1, stroke_mat.cols);
    //    PIXA *pixa = pixaRead(img_path);
    
    //    string out = string(tessearct_api.GetUTF8Text());
    
    string out = string(tessearct_api.GetUTF8Text());
    return out;

}

//rotate the inage with the Affine
cv::Mat TextDetector::ImageRotate(cv::Mat & src, const Point2f &center, double angle,double scale)
{
    
    cv::Mat M = getRotationMatrix2D(center, angle, scale);
    
    // rotate
    cv::Mat dst;
    warpAffine(src, dst, M, cvSize(src.cols, src.rows), CV_INTER_LINEAR);
    return dst;
}


//pad the image
void TextDetector::padImage(cv::Mat &image, cv::Mat &dst, Size sz, int pad_flag){
    cv::Mat img = image.clone();
    int i, j;
    
    switch (pad_flag) {
        case 0://pad up-down
            i = 2;
            j = 0;
            dst = cv::Mat::ones(Size(j * sz.width + img.cols, i * sz.height + img.rows), CV_8UC1);
            img.copyTo(cv::Mat(dst, cv::Rect(0, sz.height, img.cols, img.rows)));
            break;
        case 1://pad left-right
            i = 0;
            j = 2;
            dst = cv::Mat::ones(Size(j * sz.width + img.cols, i * sz.height + img.rows), CV_8UC1);
            img.copyTo(cv::Mat(dst, cv::Rect(sz.width, 0, img.cols, img.rows)));
            break;
        case 2://pad both
            i = 2;
            j = 2;
            dst = cv::Mat::ones(Size(j * sz.width + img.cols, i * sz.height + img.rows), CV_8UC1);
            
            img.copyTo(cv::Mat(dst, cv::Rect(sz.width, sz.height, img.cols, img.rows)));
            
        default:
            break;
    }
    
}


//Get a table of word astatus and returns a new table of words after merges all possible words
void TextDetector::mergeWords(vector<WordsStatus> &src_word_stat, cv::Mat &src_cc_dist, cv::Mat &src_cc_ang, vector<WordsStatus> &dst_word_stat, cv::Mat &dst_cc_dist, cv::Mat &dst_cc_ang){
    vector<vector<int>> words_cell_arr;
    for (int i = 0; i < src_word_stat.size(); i ++) {
        words_cell_arr.push_back(src_word_stat[i].words);
    }
    cv::Mat w_dist_mat, w_angle_mat;
    word_dist_mat(words_cell_arr, src_cc_dist, w_dist_mat);
    word_angle_mat(words_cell_arr, src_cc_dist, src_cc_ang, w_angle_mat);
    for (int k = 0; k < src_word_stat.size() - 1; k ++) {
        for (int l = k+1; l < src_word_stat.size(); l ++) {
            if(1 == checkMerge(k, l, src_word_stat, w_dist_mat, w_angle_mat)){
                vector<int> word1 = src_word_stat[k].words;
                vector<int> word2 = src_word_stat[l].words;
                vector<int> word_new;
                getMergedWord(word1, word2, src_cc_dist, word_new);
                WordsStatus word_new_stat;
                getWordsStatus(word_new, src_cc_dist, src_cc_ang, word_new_stat);
                src_word_stat[k] = word_new_stat;
                src_word_stat[l] = {};
                mergeWords(src_word_stat, src_cc_dist, src_cc_ang, dst_word_stat, dst_cc_dist, dst_cc_ang);
                
                word1.clear();
                word2.clear();
                word_new.clear();
            }
        }
    }
    
    words_cell_arr.clear();
    
    
}

//Check if two words can be merged.Return 1 if can, otherwise 0.
int TextDetector::checkMerge(int word1, int word2, vector<WordsStatus> &word_stat, cv::Mat & dist_mat, cv::Mat & angle_mat){
    int dist = dist_mat.at<int>(word1, word2);
    float angle = angle_mat.at<float>(word1, word2);
    
    float word1_dist_mean  = word_stat[word1].dist_mean;
    float word2_dist_mean  = word_stat[word2].dist_mean;
    
    float word1_dist_std   = word_stat[word1].dist_std;
    float word2_dist_std   = word_stat[word2].dist_std;
    
    float word1_angle_mean = word_stat[word1].angle_mean;
    float word2_angle_mean = word_stat[word2].angle_mean;
    
    float angle_diff = 13.0;
    float std_factor = 1.6;
    float dist_factor_1, dist_factor_2;
    if( 0 == word1_dist_std)
        dist_factor_1 = 0.6;
    else
        dist_factor_1 = 0.0;
    
    if(0 == word2_dist_std)
        dist_factor_2 = 0.6;
    else
        dist_factor_2 = 0.0;
    
    int res = 0;
    
    if (1 == word_stat[word1].length) {
        if ( angle <= (word2_angle_mean + angle_diff) && angle >= (word2_angle_mean - angle_diff)
            && dist <= (word2_dist_mean + dist_factor_2 * word2_dist_mean + std_factor * word2_dist_std)
            && dist >= (word2_dist_mean - dist_factor_2 * word2_dist_mean - std_factor * word2_dist_std) ) {
            res = 1;
//            return res;
        }
    }
    
    if (1 == word_stat[word2].length) {
        if( angle <= (word1_angle_mean + angle_diff)
           && angle >= (word1_angle_mean - angle_diff)
           && dist <= (word1_dist_mean + dist_factor_1 * word1_dist_mean + std_factor *word1_dist_std)
           && dist >= (word1_dist_mean - dist_factor_1 * word1_dist_mean - std_factor *word1_dist_std) ){
            res = 1;
//            return res;
            
        }
    }
    
    if ( ( (angle <= word1_angle_mean + angle_diff && angle >= word1_angle_mean - angle_diff)
        || (angle <= word2_angle_mean + angle_diff && angle >= word2_angle_mean - angle_diff) )
        && ( (dist <= word1_dist_mean + dist_factor_1 * word1_dist_mean + std_factor * word1_dist_std
            && dist >= word1_dist_mean - dist_factor_1 * word1_dist_mean - std_factor * word1_dist_std)
            || (dist <= word2_dist_mean + dist_factor_2 * word2_dist_mean + std_factor * word2_dist_std
                && dist >= word2_dist_mean - dist_factor_2 * word2_dist_mean - std_factor * word2_dist_std) ) ) {
                res = 1;
    }
    return res;
}

//merges word2 and word2 ,and returns the resulting word
void TextDetector::getMergedWord(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, vector<int> &merged){
    vector<int> dist_arr;
    double min;
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[word2.size()-1]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size() -1], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size()-1], word2[word2.size()-1]));
    minMaxLoc(dist_arr, &min);
    int index;
    for (int i = 0; i < dist_arr.size(); i ++) {
        if(dist_arr[i] == min)
        {
            index = i;
            break;
        }
    }
    
    if(1 == index){
        vector<int> flip_lr;
        flip(word1, flip_lr, 1);
        merged = flip_lr;
        for (int i = 0; i < word2.size(); i ++) {
            merged.push_back(word2[i]);
        }
    }
        
    else if (2 == index){
        merged = word2;
        for (int i = 0; i < word1.size(); i ++) {
            merged.push_back(word1[i]);
        }

        
    }
    else if (3 == index){
        merged = word1;
        for (int i = 0; i < word2.size(); i ++) {
            merged.push_back(word2[i]);
        }

    }
    else{
        vector<int> flip_lr;
        flip(word2, flip_lr, 1);
        merged = word1;
        for (int i = 0; i < flip_lr.size(); i ++) {
            merged.push_back(flip_lr[i]);
        }
    }

}

//find the distance between two words
void TextDetector::word_dist(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, int &dist){
    vector<int> dist_arr;
    double min;
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[word2.size()-1]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size() -1], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size()-1], word2[word2.size()-1]));
    minMaxLoc(dist_arr, &min);
    dist = (int)min;
}

//returns a matrix specifying the distances between each pair of words
void TextDetector::word_dist_mat(vector<vector<int>> &words_arr, cv::Mat &cc_dist, cv::Mat &dist_mat ){
    dist_mat = cv::Mat::zeros((int)words_arr.size(), (int)words_arr.size(), CV_8U);
    for (int k = 0; k < words_arr.size(); k ++) {
        for (int l = k+1; l < words_arr.size(); l ++) {
            vector<int> curr_word1 = words_arr[k];
            vector<int> curr_word2 = words_arr[l];
            word_dist(curr_word1, curr_word2, cc_dist, dist_mat.at<int>(k, l));
        }
    }
    cv::Mat transpose;
    cv::transpose(dist_mat, transpose);
    dist_mat = dist_mat + transpose;
    for (int i = 0; i < dist_mat.rows; i ++) {
        int *data = dist_mat.ptr<int>(i);
        for (int j = 0; j < dist_mat.cols; j ++) {
            if(i == j)
                data[j] = NaN;
        }
    }
    
}


//Find the angle between two words
void TextDetector::word_angle(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, cv::Mat &cc_angle, float &angle ){
    vector<int> dist_arr;
    double min;
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[0], word2[word2.size()-1]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size() -1], word2[0]));
    dist_arr.push_back(cc_dist.at<int>(word1[word1.size()-1], word2[word2.size()-1]));
    minMaxLoc(dist_arr, &min);
    int index;
    for (int i = 0; i < dist_arr.size(); i ++) {
        if(dist_arr[i] == min)
        {
            index = i;
            break;
        }
    }
    
    if(1 == index)
        angle = cc_angle.at<int>(word1[0], word2[0]);
    else if (2 == index)
        angle = cc_angle.at<int>(word1[0], word2[word2.size()-1]);
    else if (3 == index)
        angle = cc_angle.at<int>(word1[word1.size() -1], word2[0]);
    else
        angle = cc_angle.at<int>(word1[word1.size()-1], word2[word2.size()-1]);

}

//Returns a matrix specifying the angle between every pair of words
void TextDetector::word_angle_mat(vector<vector<int>> &words_arr, cv::Mat &cc_dist, cv::Mat &cc_angle, cv::Mat &ang_mat ){
    ang_mat = cv::Mat::zeros((int)words_arr.size(), (int)words_arr.size(), CV_32F);
    for (int k = 0; k < words_arr.size() - 1; k ++) {
        for (int l = k+1; l < words_arr.size(); l ++) {
            vector<int> curr_word1 = words_arr[k];
            vector<int> curr_word2 = words_arr[l];
            word_angle(curr_word1, curr_word2, cc_dist, cc_angle, ang_mat.at<float>(k, l));
        }
    }
    cv::Mat transpose;
    cv::transpose(ang_mat, transpose);
    ang_mat = ang_mat + transpose;
    for (int i = 0; i < ang_mat.rows; i ++) {
        int *data = ang_mat.ptr<int>(i);
        for (int j = 0; j < ang_mat.cols; j ++) {
            if(i == j)
                data[j] = NaN;
        }
    }
    
}



//get words status,return a structure
void TextDetector::getWordsStatus(vector<int> &words, cv::Mat &cc_dist, cv::Mat &cc_angle, WordsStatus &word_stat){
    vector<int> curr_dist_arr;
    get_dist_arr(words, cc_dist, curr_dist_arr);
    vector<float> curr_angle_arr;
    get_angle_array(words, cc_angle, curr_angle_arr);
    
    word_stat.words  = words;
    word_stat.length = sizeof(words);
    
    if (curr_dist_arr.size() < 2) {
        word_stat.dist_arr = {};
        word_stat.dist_mean = NaN;
        word_stat.dist_std = NaN;
    }
    else{
        word_stat.dist_arr = curr_dist_arr;
        vector<float> dist_mean, dist_std;
        meanStdDev(curr_dist_arr, dist_mean, dist_std);
        word_stat.dist_mean = dist_mean[0];
        word_stat.dist_std = dist_std[0];
        
        dist_std.clear();
        dist_mean.clear();
    }
    
    if (curr_angle_arr.size() < 2){
        word_stat.angle_arr = {};
        word_stat.angle_mean = NaN;
        word_stat.angle_std = NaN;
    }
    else{
        vector<float> angle_mean, angle_std;
        word_stat.angle_arr = curr_angle_arr;
        meanStdDev(curr_angle_arr, angle_mean, angle_std);
        word_stat.angle_mean = angle_mean[0];
        word_stat.angle_std = angle_std[0];
        
        angle_std.clear();
        angle_mean.clear();
    }
    
    curr_angle_arr.clear();
    curr_dist_arr.clear();
    
    
}

void TextDetector::get_dist_arr(vector<int> &word, cv::Mat &dist_mat, vector<int> &dist_array){
    for (int k = 0; k < word.size() - 1; k ++) {
        int curr_char = word[k];
        int next_char = word[k+1];
        dist_array.push_back(dist_mat.at<int>(curr_char, next_char));
    }
}

void TextDetector::get_angle_array(vector<int> &word, cv::Mat &angle_mat, vector<float> &angle_array){
    for (int k = 0; k < word.size() - 1; k ++) {
        int curr_char = word[k];
        int next_char = word[k+1];
        angle_array.push_back(angle_mat.at<float>(curr_char, next_char));
    }
}

//return  a matrix of angle between each two CCs
void TextDetector::cc_angle_mat(cv::Mat &cc_centers, cv::Mat &angle_mat){
    angle_mat = cv::Mat::zeros(cc_centers.rows, cc_centers.rows, CV_32F);
    for (int i = 0; i < cc_centers.rows - 1; i ++) {
        for (int j = i+1; j < cc_centers.rows; j ++) {
            Point2f tmp = cv::Point(cc_centers.at<float>(j, 0), cc_centers.at<float>(j, 1)) -
                                    cv::Point(cc_centers.at<float>(i, 0), cc_centers.at<float>(i, 1));
            float angle = abs(atan2(tmp.y, tmp.x)) * 180 / CV_PI;
            if (angle > 90)
                angle = 180 - angle;
            angle_mat.at<float>(i, j) = angle;
        }
    }
}


double TextDetector::max_array(double *a)
{
    double t = a[0];
    for(int i = 1;i < sizeof(a);i ++)
        t = (t > a[i]) ? t : a[i];
    return t;
}

double TextDetector::min_array(double *a)
{
    double t = a[0];
//    cout<<t<<endl;
    for(int i = 1;i < sizeof(a);i ++)
        
        t = (t < a[i]) ? t : a[i];
    return t;
}


//calculate the minimum Eucledian distance between two sets of pixels
void TextDetector::min_px_dist(cv::Mat &px1, cv::Mat &px2, double &dist){
    double min_dist = 9999.0;
    for (int i = 0; i < px1.rows; i ++) {
        for (int j = 0; j < px2.rows; j ++) {
            double tmp_dist = sqrt( pow((px1.at<uchar>(i, 0) - px2.at<uchar>(j, 0)), 2) +
                                 pow((px1.at<uchar>(i, 1) - px2.at<uchar>(j, 1)), 2) );
            if(tmp_dist < min_dist)
                min_dist = tmp_dist;
        }
    }
    dist = min_dist;
}

void TextDetector::findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols){
    int count = 0;
    for (int i = 0; i < data.cols; i ++) {
        uchar *u = data.ptr<uchar>(i);
        for (int j = 0; j < data.rows; j ++) {
            if(edgeValue == (int)u[j]){
                if(count < k){
                    count ++;
                    cols.push_back(i);
                    rows.push_back(j);
                }
                
            }

        }
    }
    
}

void TextDetector::findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows, vector<int> &cols){
    int count = 0;
    for (int i = data.cols - 1; i >= 0; i --) {
        uchar *u = data.ptr<uchar>(i);
        for (int j = data.rows - 1; j >= 0; j --) {
            if(edgeValue == (int)u[j]){
                if(count < k){
                    count ++;
                    cols.push_back(i);
                    rows.push_back(j);
                }
                
            }
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
    float w_a4 = sqrt(pow(in_point[0] - in_point[2], 2) + pow(in_point[1] - in_point[3] ,2));
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
    dst_pts.clear();
    img_pts.clear();
}


void TextDetector::getROI(cv::Mat &src,cv::Mat &out,cv::Rect rect)
{
    out = cv::Mat(rect.height, rect.width, CV_8UC3,Scalar(255));
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

void TextDetector::getROI2(cv::Mat &src,cv::Mat &out,cv::Rect rect){
//    out = cv::Mat(rect.width, rect.height, CV_8UC3,Scalar(255));
    vector<cv::Point2f>  quad_pts;
    //映射到原图上
    vector<cv::Point2f> pointss;
    float pt[8];
    pointss.push_back(rect.tl());
    pointss.push_back(rect.tl() + cv::Point(rect.width,0));
    pointss.push_back(rect.tl() + cv::Point(0,rect.height));
    pointss.push_back(rect.br());
    for (int i = 0; i < sizeof(pt) - 1; i = i + 2) {
        pt[i] = pointss[i].x;
        pt[i+1] = pointss[i].y;
    }
    
    pointss.clear();
    perspective(src, pt, out);

}


int TextDetector::WriteData(string fileName, cv::Mat& matData)
{
    int retVal = 0;
    ofstream outFile(fileName.c_str(), ios_base::out);
    if (!outFile.is_open())
    {
        cout << "路径不存在" << endl;
        retVal = -1;
        return (retVal);
    }
    if (matData.empty())
    {
        cout << "Mat为空’" << endl;
        retVal = 1;
        return (retVal);
    }
    for (int r = 0; r < matData.rows; r++)
    {
        outFile <<r<<"行"<<endl;
        for (int c = 0; c < matData.cols; c++)
        {
            double data= matData.at<double>(r,c);
            
            outFile <<data << "\t" ;
        }
        outFile << endl;
    }
    return (retVal);
}



void TextDetector::sharpenImage(const cv::Mat &image, cv::Mat &result)
{
    //创建并初始化滤波模板
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 5.0;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    kernel.at<float>(2,1) = -1.0;

    result.create(image.size(),image.type());

    //对图像进行滤波
    cv::filter2D(image,result,image.depth(),kernel);
}



float TextDetector::getBlobEccentricity( const Moments& moment ) {
    double left_comp  = (moment.nu20 + moment.nu02) / 2.0;
    double right_comp = sqrt( (4 * moment.nu11 * moment.nu11) + (moment.nu20 - moment.nu02)*(moment.nu20 - moment.nu02) ) / 2.0;
    
    double eig_val_1 = left_comp + right_comp;
    double eig_val_2 = left_comp - right_comp;
    
    return sqrtf( 1.0 - (eig_val_2 / eig_val_1) );
}

/**
 * From the given blob moment, calculate its centroid
 */
Point2f TextDetector::getBlobCentroid( const Moments& moment ) {
    return Point2f( moment.m10 / moment.m00, moment.m01 / moment.m00 );
}

double TextDetector::Round(double dVal, short iPlaces) {
    double dRetval;
    double dMod = 0.0000001;
    if (dVal<0.0) dMod=-0.0000001;
    dRetval=dVal;
    dRetval+=(5.0/pow(10.0,iPlaces+1.0));
    dRetval*=pow(10.0,iPlaces);
    dRetval=floor(dRetval+dMod);
    dRetval/=pow(10.0,iPlaces);
    return(dRetval);
}






































