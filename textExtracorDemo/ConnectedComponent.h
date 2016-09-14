//
//  ConnectedComponent.h
//  ConnectedComponent
//
//  Created by Saburo Okita on 06/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __RobustTextDetection__ConnectedComponent__
#define __RobustTextDetection__ConnectedComponent__

#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

/**
 * Structure that describes the property of the connected component
 */
struct ComponentProperty
{
    int labelID;//标号
    int area;//面积
    int orientation;
    float eccentricity;//离心率，偏心距
    float solidity;//置信度
    cv::Point2f centroid;//质心
    cv::Rect boxCC;
    vector<cv::Point> pixelIdxList;
    
    
    friend std::ostream &operator <<( std::ostream& os, const ComponentProperty & prop )
    {
        os << "     Label ID: " << prop.labelID      << "\n";
        os << "         Area: " << prop.area         << "\n";
        os << "     Centroid: " << prop.centroid     << "\n";
        os << " Eccentricity: " << prop.eccentricity << "\n";
        os << "     Solidity: " << prop.solidity     << "\n";
        return os;
    }
};


/**
 * Connected component labeling using 8-connected neighbors, based on
 * http://en.wikipedia.org/wiki/Connected-component_labeling
 *
 * with disjoint union and find functions adapted from :
 * https://courses.cs.washington.edu/courses/cse576/02au/homework/hw3/ConnectComponent.java
 */
class ConnectedComponent
{
public:
    ConnectedComponent( int max_component = 1000, int connectivity_type = 8 );
    virtual ~ConnectedComponent();
    
    cv::Mat apply( const cv::Mat& image );
    
    int getComponentsCount();//获取连通分量数
    const std::vector<ComponentProperty>& getComponentsProperties();//获取连通分量属性
    
    std::vector<int> get8Neighbors( int * curr_ptr, int * prev_ptr, int x );//获取8连通
    std::vector<int> get4Neighbors( int * curr_ptr, int * prev_ptr, int x );//获取4连通
    
protected:
    float calculateBlobEccentricity( const cv::Moments& moment );//计算块离心率
    cv::Point2f calculateBlobCentroid( const cv::Moments& moment );//计算块质心
    
    void disjointUnion( int a, int b, std::vector<int>& parent  );//计算不相交的区域
    int disjointFind( int a, std::vector<int>& parent, std::vector<int>& labels  );//查找不相交得块
    
private:
    int connectivityType;//连通域类型
    int maxComponent;//最大连通面积
    int nextLabel;//下一个标记
    std::vector<ComponentProperty> properties;//每一个分量的属性
};

#endif /* defined(__RobustTextDetection__ConnectedComponent__) */
