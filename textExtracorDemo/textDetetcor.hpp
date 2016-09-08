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
#include <fstream>


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


struct ConnCompStats {
    cv::Rect bboxes;
    int bbox_width;
    int bbox_height;
    double bbox_aspect;
    int bbox_area;
    vector<cv::Point2f> bbox_idx_list;
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
    void getROI2(cv::Mat &src,cv::Mat &out,cv::Rect rect);
    void findKEdgeFirst(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
    void findKEdgeLast(cv::Mat &data, int edgeValue,int k,vector<int> &rows,vector<int> &cols);
    void min_px_dist(vector<cv::Point> &px1, vector<cv::Point> &px2, int &dist);
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
    int WriteData(string fileName, cv::Mat& matData);
    void sharpenImage(const cv::Mat &image, cv::Mat &result);
    void imgQuantize(cv::Mat &src, cv::Mat &dst, double level);
    float getBlobEccentricity( const Moments& moment ) ;
    Point2f getBlobCentroid( const Moments& moment ) ;
private:
    string imageDirectory;
    TextDetecorParams Detectorparams;
};  /*  class TextDetector */






class Histogrom1D
{
private:
    int histSize[1];//项的数量
    float hranges[2];//像素的最大与最小值
    const float *ranges[1];
    int channels[1];//仅用到一个通道
public:
    Histogrom1D(){
        //准备1D直方图的参数
        histSize[0] = 256;
        hranges[0] = 0.0;
        hranges[1] = 255.0;
        ranges[0] = hranges;
        channels[0] = 0;//默认情况，我们考察0号通道
        
    }
    //计算一维直方图
    MatND getHistogram(const cv::Mat &image){
        cv::MatND hist;
        calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
        return hist;
    }
    //计算一维直方图，并返回一幅图像
    cv::Mat getHistogramImage(const cv::Mat &image,double &mean)
    {
        MatND hist = getHistogram(image);
        double maxVal = 0,minVal = 0;
        //获取最大值和最小值
        minMaxLoc(hist, &minVal, &maxVal,0,0);
        cout<<maxVal<<endl<<minVal<<endl;
        cv::Mat histImg(histSize[0],histSize[0],CV_8U,Scalar(255));
        //设置最高点为nbins的90%
        int hpt = static_cast<int>(0.9 * histSize[0]);
        long sum = 0;
        //每个条目绘制一条垂直线
        for(int h = 0;h < histSize[0];h ++)
        {
            float binVal = hist.at<float>(h);
            //cout<<binVal<<endl;
            int intensity = static_cast<int>(binVal*hpt/maxVal);
            sum = sum + intensity;
            line(histImg, cv::Point(h,histSize[0]), cv::Point(h,histSize[0]-intensity), Scalar::all(0));
            
        }
        mean = sum / 256;
        return histImg;
    }
    
    cv::Mat applyLookUp(const cv::Mat &image,cv::Mat &lookup)
    {
        cv::Mat result;
        cv::LUT(image, lookup, result);
        return result;
        
    }
    cv::Mat stretch(const cv::Mat &image,int minValue)
    {
        //首先计算直方图
        MatND hist = getHistogram(image);
        //寻找直方图的左端
        int imin = 0;
        for (; imin < histSize[0]; imin++)
        {
            //cout<<"imin"<<hist.at<float>(imin)<<endl;
            if (hist.at<float>(imin) > minValue)break;
            else
            {
                hist.at<float>(imin) = minValue;
            }
        }
        //寻找直方图的右端
        int imax = histSize[0] - 1;
        for (; imax >= 0; imax --)
        {
            if (hist.at<float>(imax) > minValue)
            {
                break;
            }
            else
                hist.at<float>(imax) = 255;
        }
        //创建查找表
        int dim(256);
        cv::Mat lookup(1,&dim,CV_8U);
        //填充查找表
        for (int i = 0; i < 256; i ++)
        {
            //确保数值位于imin和imax之间
            if (i < imin)lookup.at<uchar>(i) = 0;
            else if (i > imax)lookup.at<uchar>(i) = 255;
            //线性映射
            else
                lookup.at<uchar>(i) = static_cast<uchar>(255.0 * (i - imin)/(imax - imin) + 0.5);
        }
        //应用查找表
        cv::Mat res;
        res = applyLookUp(image, lookup);
        
        return res;
        
        
    }
    
    void getKValue(Mat &src, int &mean,int &kMax)
    {
        Mat gray;
        cvtColor(src, gray, CV_BGR2GRAY);
        MatND hist = getHistogram(gray);
        double maxVal = 0,minVal = 0;
        //        int k_min,k_max;
        //获取最大值和最小值
        minMaxLoc(hist, &minVal, &maxVal,0,0);
        int hpt = static_cast<int>(0.9 * histSize[0]);
        long sum = 0;
        //每个条目绘制一条垂直线
        for(int h = 0;h < histSize[0];h ++)
        {
            float binVal = hist.at<float>(h);
            //cout<<binVal<<endl;
            int intensity = static_cast<int>(binVal*hpt/maxVal);
            sum = sum + intensity;
            if (hist.at<float>(h) == maxVal)
            {
                kMax = h;
            }
        }
        mean = (int)(sum / 256);
        
    }
    void getMeanVar(Mat &src,double &mean,double &var)
    {
        int width = src.cols;
        int height = src.rows;
        
        uchar *ptr = src.data;
        int Iij;
        
        double Imax = 0, Imin = 255, Iave = 0, Idelta = 0;
        
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {
                Iij	= (int) ptr[i*width+j];
                
                if(Iij > Imax)
                    Imax = Iij;
                
                if(Iij < Imin)
                    Imin = Iij;
                
                Iave = Iave + Iij;
            }
        }
        
        Iave = Iave/(width*height);
        mean = Iave;
        
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {
                Iij	= (int) ptr[i*width+j];
                
                Idelta	= Idelta + (Iij-Iave)*(Iij-Iave);
            }
        }
        
        Idelta = Idelta/(width*height);
        var = sqrt(Idelta);
    }
    
    void changeRGB(Mat &src)
    {
        //        Mat image;
        //        detailEnhance(src, src);
        Mat gray;
        cvtColor(src, gray, CV_BGR2GRAY);
        MatND hist = getHistogram(gray);
        double maxVal = 0,minVal = 0;
        //获取最大值和最小值
        int hpt = static_cast<int>(0.9 * histSize[0]);
        minMaxLoc(hist, &minVal, &maxVal,0,0);
        cout<<minVal<<","<<maxVal<<endl;
        vector<int> kChange;
        //        int th = me / maxVal;
        //        int nonzero = countNonZero(gray);
        //        double ratio = nonzero / (double)(gray.cols * gray.rows);
        //        cout<<ratio<<endl;
        for (int i = 0; i < histSize[0]; i ++)
        {
            float binVal = hist.at<float>(i);
            //cout<<binVal<<endl;
            int intensity = static_cast<int>(binVal*hpt/maxVal);
            if (abs(intensity - 255) < 60)
            {
                intensity = 255;
                hist.at<float>(i) = intensity * maxVal/hpt;
            }
            //            if (abs(intensity - 0) > 30) {
            //                intensity = 0;
            //                hist.at<float>(i) = intensity * maxVal/hpt;
            //            }
            //
            
        }
        minMaxLoc(hist,&minVal, &maxVal,0,0);
        cout<<minVal<<","<<maxVal<<endl;
        for (int num = 0; num < histSize[0]; num ++)
        {
            if (hist.at<float>(num) >= 0.5*maxVal)//( hist.at<float>(i) >= ratio*maxVal && hist.at<float>(i) <= 1.0*maxVal)//3.63
            {
                kChange.push_back(num);
            }
        }
        int nl = src.rows;
        int nc = src.cols;
        //        if (src.isContinuous()) {
        //            nc = nc * nl;
        //            nl = 1;
        //        }
        for (int j = 0; j < nl; j ++)
        {
            //            uchar *data = src.ptr<uchar>(j);
            for (int l = 0; l < nc; l ++)
            {
                int b = 0.11 * src.at<Vec3b>(j, l)[0];
                int g = 0.59 * src.at<Vec3b>(j, l)[1];
                int r = 0.30 * src.at<Vec3b>(j, l)[2];
                int greyVal = b + g + r;
                for (int ii = 0; ii < kChange.size(); ii ++)
                {
                    if (abs(greyVal - kChange[ii]) <= 38)//
                        //                    if (greyVal == kChange[ii])
                    {
                        src.at<Vec3b>(j, l)[0] = 255;
                        src.at<Vec3b>(j, l)[1] = 255;
                        src.at<Vec3b>(j, l)[2] = 255;
                    }
                }
                
            }
        }
    }
    
    Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
    {
        // accept only char type matrices
        CV_Assert(I.depth() != sizeof(uchar));
        int channels = I.channels();
        int nRows = I.rows ;
        int nCols = I.cols* channels;
        if (I.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        int i,j;
        uchar* p;
        for( i = 0; i < nRows; ++i)
        {
            p = I.ptr<uchar>(i);
            for ( j = 0; j < nCols; ++j)
            {
                p[j] = table[p[j]];
            }
        }
        return I;
    }
    
    Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
    {
        // accept only char type matrices
        CV_Assert(I.depth() != sizeof(uchar));
        const int channels = I.channels();
        switch(channels)
        {
            case 1:
            {
                MatIterator_<uchar> it, end;
                for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                    *it = table[*it];
                break;
            }
            case 3:
            {
                MatIterator_<Vec3b> it, end;
                for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
                {
                    (*it)[0] = table[(*it)[0]];
                    (*it)[1] = table[(*it)[1]];
                    (*it)[2] = table[(*it)[2]];
                }
            }
        }
        return I;
    }
    
    
    //方法零：.ptr和[]操作符
    //Mat最直接的访问方法是通过.ptr<>函数得到一行的指针，并用[]操作符访问某一列的像素值。
    void colorReduce0(cv::Mat &image, int div) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                data[i]= data[i]/div*div + div/2;
            }
        }
    }
    
    
    void grayIncrease(cv::Mat &image,int grayVal) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                data[i]= data[i] - grayVal;
            }
        }
    }
    
    //方法一：.ptr和指针操作
    //除了[]操作符，我们可以移动指针*++的组合方法访问某一行中所有像素的值。
    // using .ptr and * ++
    void colorReduce1(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                uchar * tmp;
                *tmp ++ = *data ++;
                *tmp++ = *data/div*div + div/2;
            } // end of row
        }
    }
    
    //方法二：.ptr、指针操作和取模运算
    //方法二和方法一的访问方式相同，不同的是color reduce用模运算代替整数除法
    // using .ptr and * ++ and modulo
    void colorReduce2(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                int v= *data;
                *data++= v - v%div + div/2;
            } // end of row
        }
    }
    
    //方法三：.ptr、指针运算和位运算
    //由于进行量化的单元div通常是2的整次方，因此所有的乘法和除法都可以用位运算表示。
    
    // using .ptr and * ++ and bitwise
    void colorReduce3(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                uchar * tmp;
                *tmp ++ = *data ++;
                *tmp++= *data&mask + div/2;
            } // end of row
        }
    }
    
    //方法四：指针运算
    ///方法四和方法三量化处理的方法相同，不同的是用指针运算代替*++操作。
    
    // direct pointer arithmetic
    void colorReduce4(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        int step= (int)image.step; // effective width
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        // get the pointer to the image buffer
        uchar *data= image.data;
        for (int j=0; j<nr; j++) {
            for (int i=0; i<nc; i++) {
                *(data+i)= *data&mask + div/2;
            } // end of row
            data+= step;  // next line
        }
    }
    
    //方法五：.ptr、*++、位运算以及image.cols * image.channels()
    //这种方法就是没有计算nc，基本是个充数的方法。
    
    // using .ptr and * ++ and bitwise with image.cols * image.channels()
    void colorReduce5(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<image.cols * image.channels(); i++) {
                uchar * tmp;
                *tmp ++ = *data ++;
                *tmp++= *data&mask + div/2;
            } // end of row
        }
    }
    
    
    //方法六：连续图像
    //Mat提供了isContinuous()函数用来查看Mat在内存中是不是连续存储，如果是则图片被存储在一行中。
    
    // using .ptr and * ++ and bitwise (continuous)
    void colorReduce6(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels(); // total number of elements per line
        if (image.isContinuous())  {
            // then no padded pixels
            nc= nc*nr;
            nr= 1;  // it is now a 1D array
        }
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                uchar * tmp;
                *tmp ++ = *data ++;
                *tmp++= *data&mask + div/2;
            } // end of row
        }
    }
    
    //方法七：continuous+channels
    //与方法六基本相同，也是充数的。
    
    // using .ptr and * ++ and bitwise (continuous+channels)
    void colorReduce7(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols ; // number of columns
        if (image.isContinuous())  {
            // then no padded pixels
            nc= nc*nr;
            nr= 1;  // it is now a 1D array
        }
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        for (int j=0; j<nr; j++) {
            uchar* data= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                uchar * tmp;
                *tmp ++ = *data ++;
                *tmp++= *data&mask + div/2;
                *tmp++= *data&mask + div/2;
                *tmp++= *data&mask + div/2;
            } // end of row
        }
    }
    
    //方法八：Mat _iterator
    //真正有区别的方法来啦，用Mat提供的迭代器代替前面的[]操作符或指针，血统纯正的官方方法~
    
    // using Mat_ iterator
    void colorReduce8(cv::Mat &image, int div=64) {
        // get iterators
        cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();
        for ( ; it!= itend; ++it) {
            (*it)[0]= (*it)[0]/div*div + div/2;
            (*it)[1]= (*it)[1]/div*div + div/2;
            (*it)[2]= (*it)[2]/div*div + div/2;
        }
    }
    
    
    //方法九：Mat_ iterator 和位运算
    //把方法八中的乘除法换成位运算。
    
    // using Mat_ iterator and bitwise
    void colorReduce9(cv::Mat &image, int div=64) {
        // div must be a power of 2
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        // get iterators
        cv::Mat_<cv::Vec3b>::iterator it= image.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= image.end<cv::Vec3b>();
        for ( ; it!= itend; ++it) {
            (*it)[0]= (*it)[0]&mask + div/2;
            (*it)[1]= (*it)[1]&mask + div/2;
            (*it)[2]= (*it)[2]&mask + div/2;
        }
    }
    
    //方法十：MatIterator_
    //和方法八基本相同。
    
    // using MatIterator_
    void colorReduce10(cv::Mat &image, int div=64) {
        cv::Mat_<cv::Vec3b> cimage= image;
        cv::Mat_<cv::Vec3b>::iterator it=cimage.begin();
        cv::Mat_<cv::Vec3b>::iterator itend=cimage.end();
        for ( ; it!= itend; it++) {
            (*it)[0]= (*it)[0]/div*div + div/2;
            (*it)[1]= (*it)[1]/div*div + div/2;
            (*it)[2]= (*it)[2]/div*div + div/2;
        }
    }
    
    
    //方法十一：图像坐标
    
    // using (j,i)
    void colorReduce11(cv::Mat &image, int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols; // number of columns
        for (int j=0; j<nr; j++) {
            for (int i=0; i<nc; i++) {
                image.at<cv::Vec3b>(j,i)[0]=     image.at<cv::Vec3b>(j,i)[0]/div*div + div/2;
                image.at<cv::Vec3b>(j,i)[1]=     image.at<cv::Vec3b>(j,i)[1]/div*div + div/2;
                image.at<cv::Vec3b>(j,i)[2]=     image.at<cv::Vec3b>(j,i)[2]/div*div + div/2;
            } // end of row
        }
    }
    
    //方法十二：创建输出图像
    //之前的方法都是直接修改原图，方法十二新建了输出图像，主要用于后面的时间对比。
    
    // with input/ouput images
    void colorReduce12(const cv::Mat &image, // input image
                       cv::Mat &result,      // output image
                       int div=64) {
        int nr= image.rows; // number of rows
        int nc= image.cols ; // number of columns
        // allocate output image if necessary
        result.create(image.rows,image.cols,image.type());
        // created images have no padded pixels
        nc= nc*nr;
        nr= 1;  // it is now a 1D array
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        for (int j=0; j<nr; j++) {
            uchar* data= result.ptr<uchar>(j);
            const uchar* idata= image.ptr<uchar>(j);
            for (int i=0; i<nc; i++) {
                *data++= (*idata++)&mask + div/2;
                *data++= (*idata++)&mask + div/2;
                *data++= (*idata++)&mask + div/2;
            } // end of row
        }
    }
    
    //方法十三：重载操作符
    //Mat重载了+&等操作符，可以直接将两个Scalar(B,G,R)数据进行位运算和数学运算。
    
    // using overloaded operators
    void colorReduce13(cv::Mat &image, int div=64) {
        int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
        // mask used to round the pixel value
        uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0
        // perform color reduction
        image=(image&cv::Scalar(mask,mask,mask))+cv::Scalar(div/2,div/2,div/2);
    }
    //指针*++访问和位运算是最快的方法；而不断的计算image.cols*image.channles()花费了大量重复的时间；另外迭代器访问虽然安全，但性能远低于指针运算；通过图像坐标(j,i)访问时最慢的，使用重载操作符直接运算效率最高。
    
    
    
    
    
};





#endif /* textDetetcor_hpp */
