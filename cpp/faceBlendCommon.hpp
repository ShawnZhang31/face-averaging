/**
 * 
 * @file faceBlendCommon.hpp
 * @author Shawn Zhang
 * @date 2018-09-09
 */
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159
#endif

/**
 * @brief 将点约束在图像的内部
 * 
 * @param p 点坐标
 * @param size 图像尺寸
 */
void constrainPoint(cv::Point2f &p, cv::Size size)
{
    p.x = min(max((double)p.x, 0.0), (double)(size.width - 1));
    p.y = min(max((double)p.y, 0.0), (double)(size.height - 1));
}

/**
 * @brief 获取顶点的索引
 * 
 * @param points 顶点集合
 * @param point 点
 * @return int 返回索引值
 */
static int findIndex(vector<Point2f>& points, Point2f &point)
{
    int minIndex = 0;
    double minDistance = norm(points[0] - point);
    for(int i=0; i<points.size();i++)
    {
        double distance = norm(points[i]-point);
        if(distance < minDistance)
        {
            minIndex = i;
            minDistance = distance;
        }
    }
    return minIndex;
}

/**
 * @brief 在图片边缘获取八个边界点
 * 
 * @param size 
 * @param boundaryPts 
 */
void getEightBoundaryPoints(Size size, vector<Point2f> &boundaryPts)
{
    int h = size.height, w = size.width;
    boundaryPts.push_back(Point2f(0, 0));
    boundaryPts.push_back(Point2f(w / 2, 0));
    boundaryPts.push_back(Point2f(w - 1, 0));
    boundaryPts.push_back(Point2f(w - 1, h / 2));
    boundaryPts.push_back(Point2f(w - 1, h - 1));
    boundaryPts.push_back(Point2f(w / 2, h - 1));
    boundaryPts.push_back(Point2f(0, h - 1));
    boundaryPts.push_back(Point2f(0, h / 2));
}

/**
 * @brief 计算相似变换矩阵
 * 
 * @param inPoints 变换前的点
 * @param outPoints 变换后的点
 * @param tfrom 相似变换矩阵
 */
void similarityTransform(std::vector<cv::Point2f> &inPoints, std::vector<cv::Point2f> &outPoints, cv::Mat &tfrom)
{
    double s60 = sin(60 * M_PI / 180.0);
    double c60 = cos(60 * M_PI / 180.0);

    std::vector<cv::Point2f> inPts = inPoints;
    std::vector<cv::Point2f> outPts = outPoints;

    // opencv计算相似变换矩阵的时候需要3个点
    inPts.push_back(cv::Point2f(0.0, 0.0));
    outPts.push_back(cv::Point2f(0.0, 0.0));

    // 这里假设给出的两个点是水平的，第三个点与给出的两个点构成等边三角形，所有可以计算第三个点；这里的计算方法使用的是矩阵变换的方法
    inPts[2].x = c60 * (inPts[1].x - inPts[0].x) + s60 * (inPts[1].y - inPts[0].y) + inPts[0].x;
    inPts[2].y = c60 * (inPts[1].y - inPts[0].y) - s60 * (inPts[1].x - inPts[0].x) + inPts[0].y;

    outPts[2].x = c60 * (outPts[1].x - outPts[0].x) + s60 * (outPts[1].y - outPts[0].y) + outPts[0].x;
    outPts[2].y = c60 * (outPts[1].y - outPts[0].y) - s60 * (outPts[1].x - outPts[0].x) + outPts[0].y;

    tfrom = cv::estimateRigidTransform(inPts, outPts, false);
}

/**
 * @brief 使用指定的尺寸规范化图片
 * 使用dlib检测到的脸部关键特征点做为输入点，使用两眼的外眼角进行规范化计算。
 * 图片规范化之后，左眼的外眼角在输出图像的坐标是(0.3*w,h/3),右眼的外眼角的坐标是(0.7*w,h/3)。w和h分别表示输出图像的宽和高
 * 
 * @param outSize 输出的图片尺寸
 * @param imgIn 输入的图片
 * @param imgOut 输出的图片
 * @param pointsIn 输入的点
 * @param pointsOut 输出的点
 */
void normalizeImagesAndLandmarks(cv::Size outSize, cv::Mat &imgIn, cv::Mat &imgOut, std::vector<cv::Point2f> &pointsIn, std::vector<cv::Point2f> &pointsOut)
{
    int h = outSize.height;
    int w = outSize.width;

    std::vector<Point2f> eyecornerSrc;
    // 左眼角
    eyecornerSrc.push_back(pointsIn[36]);
    // 右眼角
    eyecornerSrc.push_back(pointsIn[45]);

    std::vector<Point2f> eyecornerDst;
    eyecornerDst.push_back(cv::Point2f(0.3 * w, h / 3.0));
    eyecornerDst.push_back(cv::Point2f(0.7 * w, h / 3.0));

    // 计算相似变换矩阵
    cv::Mat tform;
    similarityTransform(eyecornerSrc, eyecornerDst, tform);

    // 将相似变换矩阵用在输出图像上
    imgOut = Mat::zeros(h, w, CV_32FC3);
    cv::warpAffine(imgIn, imgOut, tform, imgOut.size());

    cv::transform(pointsIn, pointsOut, tform);
}

/**
 * @brief 对图像使用仿射变换
 * 
 * @param warpImage 
 * @param src 
 * @param srcTri 
 * @param dstTri 
 */
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
    warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
}

/**
 * @brief 变换图像中的三角区域
 * 
 * @param imgIn 
 * @param imgOut 
 * @param tin 
 * @param tout 
 */
void warpTriangle(cv::Mat &imgIn, cv::Mat &imgOut, std::vector<cv::Point2f> tin, std::vector<cv::Point2f> tout)
{
    imshow("imgIn", imgIn);
    imshow("imgOut-be", imgOut);
    // 计算包围盒
    cv::Rect rect1 = cv::boundingRect(tin);
    cv::Rect rect2 = cv::boundingRect(tout);

    std::vector<cv::Point2f> tinRect, toutRect;
    std::vector<cv::Point> toutRectInt;
    for (int i = 0; i < 3; i++)
    {
        toutRectInt.push_back(cv::Point((int)(tout[i].x - rect2.x), (int)(tout[i].y - rect2.y)));

        tinRect.push_back(cv::Point2f(tin[i].x - rect1.x, tin[i].y - rect1.y));
        toutRect.push_back(cv::Point2f(tout[i].x - rect2.x, tout[i].y - rect2.y));
    }

    // Mask
    cv::Mat mask = cv::Mat::zeros(rect2.height, rect2.width, CV_32FC3);
    cv::fillConvexPoly(mask, toutRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

    cv::Mat imgInRect, imgOutRect;
    imgIn(rect1).copyTo(imgInRect);

    cv::Mat warpImage = cv::Mat::zeros(rect2.height, rect2.width, imgInRect.type());

    applyAffineTransform(warpImage, imgInRect, tinRect, toutRect);

    cv::multiply(warpImage, mask, warpImage);
    cv::multiply(imgOut(rect2), cv::Scalar(1.0, 1.0, 1.0) - mask, imgOut(rect2));

    imgOut(rect2) = imgOut(rect2) + warpImage;
    cv::imshow("imgOut", imgOut);
    cv::waitKey(100);
}

/**
 * @brief 按照Delaunay剖分法变换图像
 * 
 * @param imgIn 
 * @param imgOut 
 * @param pointsIn 
 * @param pointsOut 
 * @param delaunayTri 
 */
void warpImage(cv::Mat &imgIn, cv::Mat &imgOut, std::vector<cv::Point2f> &pointsIn, std::vector<cv::Point2f> &pointsOut, std::vector<std::vector<int>> &delaunayTri)
{
    // 输出图像和输出图像的尺寸应该是相同的
    cv::Size size = imgIn.size();
    imgOut = cv::Mat::zeros(size, imgIn.type());

    // 将delaunay剖分出的三角形映射到每个输出三角形中
    for (size_t j = 0; j < delaunayTri.size(); j++)
    {
        std::vector<Point2f> tin, tout;

        for (int k = 0; k < 3; k++)
        {
            cv::Point2f pIn = pointsIn[delaunayTri[j][k]];
            // 确保点在图像内部
            constrainPoint(pIn, size);

            cv::Point2f pOut = pointsOut[delaunayTri[j][k]];
            // 确保点在图像内部
            constrainPoint(pOut, size);

            tin.push_back(pIn);
            tout.push_back(pOut);

            

        }

        // cv::line(imgIn, tin[0], tin[1], Scalar(1.0,1.0,1.0),2);
        // cv::line(imgIn, tin[1], tin[2], Scalar(1.0,1.0,1.0),2);
        // cv::line(imgIn, tin[2], tin[0], Scalar(1.0,1.0,1.0),2);

        // cv::line(imgOut, tout[0], tout[1], Scalar(1.0,1.0,1.0),2);
        // cv::line(imgOut, tout[1], tout[2], Scalar(1.0,1.0,1.0),2);
        // cv::line(imgOut, tout[2], tout[0], Scalar(1.0,1.0,1.0),2);
        warpTriangle(imgIn, imgOut, tin, tout);
    }

}

/**
 * @brief 从detection提出关键点坐标
 * 
 * @param landmarks 
 * @param points 
 */
void dlibLandmarksToPoints( dlib::full_object_detection &landmarks, std::vector<cv::Point2f> &points)
{
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
        cv::Point2f p(landmarks.part(i).x(), landmarks.part(i).y());
        points.push_back(p);
    }
}

/**
 * @brief 对比区域大小
 * 
 * @param r1 
 * @param r2 
 * @return true 
 * @return false 
 */
bool rectAreaComparator(dlib::rectangle &r1, dlib::rectangle &r2)
{
    return r1.area() < r2.area();
}

/**
 * @brief 获取图像的面部关键点
 * 
 * @param faceDetector 
 * @param landmarkDetector 
 * @param img 
 * @param FACE_DOWNSAMPLE_RATIO 
 * @return std::vector<cv::Point2f> 
 */
std::vector<cv::Point2f> getLandmarks(dlib::frontal_face_detector &faceDetector, dlib::shape_predictor &landmarkDetector, cv::Mat &img, float FACE_DOWNSAMPLE_RATIO = 1)
{
    std::vector<cv::Point2f> points;
    cv::Mat imgSmall;
    cv::resize(img, imgSmall, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

    // 将图像转化为dlib格式
    dlib::cv_image<dlib::bgr_pixel> dlibIm(img);
    dlib::cv_image<dlib::bgr_pixel> dlibImSmall(imgSmall);

    std::vector<dlib::rectangle> faceRects = faceDetector(dlibImSmall);

    if (faceRects.size() > 0)
    {
        // 只取最大的脸
        dlib::rectangle rect = *std::max_element(faceRects.begin(), faceRects.end(), rectAreaComparator);

        dlib::rectangle scaleRect(
            (long)(rect.left() * FACE_DOWNSAMPLE_RATIO),
            (long)(rect.top() * FACE_DOWNSAMPLE_RATIO),
            (long)(rect.right() * FACE_DOWNSAMPLE_RATIO),
            (long)(rect.bottom() * FACE_DOWNSAMPLE_RATIO));

        dlib::full_object_detection landmarks = landmarkDetector(dlibIm, scaleRect);
        dlibLandmarksToPoints(landmarks, points);
    }

    return points;
}


/**
 * @brief 计算Delaunay三角形
 * 
 * @param rect 
 * @param points 
 * @param delaunayTri 
 */
void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector<std::vector<int>> &delaunayTri)
{
    Subdiv2D subdiv(rect);
    for( std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }

    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

     vector<Point2f> pt(3);
  
    vector<int> ind(3);
  
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
    
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5]);
    
    
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            for(int j = 0; j < 3; j++)
            {
                ind[j] = findIndex(points, pt[j]);
            }
            delaunayTri.push_back(ind);
        }
    } 
}