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
#include <tesseract/unichar.h>
#include <tesseract/unicharset.h>
#include <leptonica/allheaders.h>


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

struct WordsStatus {
    vector<int> words;
    int length;
    vector<int> dist_arr;
    vector<float> angle_arr;
    float dist_mean;
    float dist_std;
    float angle_mean;
    float angle_std;
    
    
};


class TextDetector{
public:
    TextDetector(string imgDir = "");
    TextDetector(TextDetecorParams &params, string imgDir = "");
    
    pair<cv::Mat, cv::Rect> applyTo(cv::Mat &image);
    
    void segmentText(cv::Mat &spineImage, cv::Mat &segSpine, bool removeNoise);
    void findWords(cv::Mat &seg_spine, int mergeFlag, cv::Mat &w_spine, vector<WordsStatus> &words_status);
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
    
    void adaptiveHistEqual(cv::Mat &src,cv::Mat &dst,double clipLimit);
    void perspective(Mat &src, float in_point[8], Mat &dst);
    void getROI(cv::Mat &src,cv::Mat &out,cv::Rect rect);
    void findKEdge(uchar *data, int edgeValue,int k,vector<int> &coords);
    
    void min_px_dist(vector<Point2f> &px1, vector<Point2f> &px2, int &dist);
    int max_array(int *a);
    int min_array(int *a);
    void cc_angle_mat(cv::Mat &cc_centers, cv::Mat &angle_mat);
    void getWordsStatus(vector<int> &words, cv::Mat &cc_dist, cv::Mat &cc_angle, WordsStatus &word_stat);
    void get_dist_arr(vector<int> &word, cv::Mat &dist_mat, vector<int> &dist_array);
    void get_angle_array(vector<int> &word, cv::Mat &angle_mat, vector<float> &angle_array);
    void mergeWords(vector<WordsStatus> &src_word_stat, cv::Mat &src_cc_dist, cv::Mat &src_cc_ang, vector<WordsStatus> &dst_word_stat, cv::Mat &dst_cc_dist, cv::Mat &dst_cc_ang);
    void word_dist(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, int &dist);
    void word_dist_mat(vector<vector<int>> &words_arr, cv::Mat &cc_dist, cv::Mat &dist_mat );
    void word_angle(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, cv::Mat &cc_angle, float &angle );
    void word_angle_mat(vector<vector<int>> &words_arr, cv::Mat &cc_dist, cv::Mat &cc_angle, cv::Mat &ang_mat );
    void getMergedWord(vector<int> &word1, vector<int> &word2, cv::Mat &cc_dist, vector<int> &merged);
    int checkMerge(int word1, int word2, vector<WordsStatus> &word_stat, cv::Mat & dist_mat, cv::Mat & angle_mat);
    
private:
    string imageDirectory;
    TextDetecorParams Detectorparams;
};  /*  class TextDetector */







#endif /* textDetetcor_hpp */
