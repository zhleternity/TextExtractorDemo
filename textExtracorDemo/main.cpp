//
//  main.cpp
//  textExtracorDemo
//
//  Created by lingLong on 16/8/29.
//  Copyright © 2016年 ling. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "textDetetcor.hpp"
#include "ConnectedComponent.h"


using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    // insert code here...
//    cv::Mat mat(3,2,CV_8UC1);
//    mat.at<int>(0,0) = 1;
//    mat.at<int>(0,1) = 2;
//    mat.at<int>(1,0) = 1;
//    mat.at<int>(1,1) = 4;
//    mat.at<int>(2,0) = 1;
//    mat.at<int>(2,1) = 1;
//    cv::Mat dst = (mat == 1);
//    for (int i = 0; i < dst.rows; i ++) {
//        for (int j = 0; j < dst.cols; j ++) {
//            std::cout<<dst.at<int>(i,j)<<std::endl;
//        }
//    }
//    std::cout << "Hello, World!\n";
    cv::Mat image = imread("/Users/eternity/Documents/test/textExtracorDemo/1.jpg");//("/Users/eternity/Documents/study/Identification of Spine(new)/query/book15.jpg");
    
    TextDetecorParams params;
    params.minMSERArea = 1;
    params.maxMSERArea = 2000;
    params.cannyThresh1 = 20;
    params.cannyThresh2 = 100;
    
    params.maxConnComponentNum = 3000;
    params.minConnComponentArea = 5;
    params.maxConnComponentArea = 600;
    
    params.minEccentricity = 0.1;
    params.maxEccentricity = 0.995;
    params.minSolidity = 0.4;
    params.maxStdDevMeanRatio = 0.7;
    
    
    string out_save_path = "/Users/eternity/Documents/test/textExtracorDemo/out";
    TextDetector detector(params, out_save_path);
    pair<cv::Mat, cv::Rect> result = detector.applyTo(image);
    imshow("result", result.first);
    
    //get the candidate text region
    cv::Mat stroke_width(result.second.height, result.second.width, CV_8UC1, Scalar(0));
    cv::Mat(result.first, result.second).copyTo(stroke_width);
    const char *img_path = "/Users/eternity/Documents/test/textExtracorDemo/out/stroke_width.jpg";
    imwrite(img_path, stroke_width);
    
    //use Tesseract to decipher the image
    double t = getTickCount();
    tesseract::TessBaseAPI tessearct_api;
    const char  *languagePath = "/usr/local/Cellar/tesseract/3.04.01_2/share/tessdata";
    const char *languageType = "chi";
    int nRet = tessearct_api.Init(languagePath, languageType,tesseract::OEM_DEFAULT);
    if (nRet != 0) {
        printf("初始化字库失败！");
        return -1;
    }
    tessearct_api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tessearct_api.SetImage(stroke_width.data, stroke_width.cols, stroke_width.rows, 1, stroke_width.cols);
    PIXA *pixa = pixaRead(img_path);

    string out = string(tessearct_api.GetWords((Pixa *)pixa));
    cout<<"the out result :"<<out<<endl;
    //split the string by whitespace
    vector<string> split;
    istringstream iss(out);
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(split));
    
    
    //show
//    CvFont font = cvFontQt("Helvetica", 24.0, CV_RGB(0, 0, 0));
    
//    CvFont font2 = cvFont(24.0);
    QtFont font2 = fontQt("Helvetica", 24.0, CV_RGB(0, 0, 0));
    cv::Point pnt = cv::Point(result.second.br().x + 1, result.second.tl().y);
    for(string &line : split ){
        addText(image, line, pnt, font2);
        
        pnt.y += 25;
    }
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout<<"It consumes:"<<t<<"second"<<endl;
    ml::KNearest *knn;
//    knn->findNearest(<#InputArray samples#>, <#int k#>, <#OutputArray results#>)
    rectangle(image, result.second, Scalar(0, 255, 0), 2);
    
    //append the original and stroke width images together
    cvtColor(stroke_width, stroke_width, CV_GRAY2BGR);
    cv::Mat append(image.rows, image.cols + stroke_width.cols, CV_8UC3);
    image.copyTo(cv::Mat(append, cv::Rect(0,0, image.cols, image.rows)));
    stroke_width.copyTo(cv::Mat(append, cv::Rect(image.cols, 0, stroke_width.cols, stroke_width.rows)));
    
    imshow("appended", append);
    waitKey();
    
    
    return 0;
}
