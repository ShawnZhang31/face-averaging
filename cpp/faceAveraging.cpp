#include "faceBlendCommon.hpp"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <vector>

// 在linux或者Unix系统中dirent.h文件并不是包含的，但是在Windows上是与包含的，所以我们分开指定
#ifdef _WIN32
#include "dirent.h"
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#include <dirent.h>
#else
#error "非Mac系统"
#endif
#elif __linux__
#include <dirent.h>
#elif __unix__
#include <dirent.h>
#else
#error "我真的不知道你的系统了"
#endif

void readFileName(string dirName, vector<string> &imageFnames)
{
    DIR *dir;
    struct dirent *ent;
    int count = 0;

    string imgExt = "jpg";
    vector<string> files;

    if ((dir = opendir(dirName.c_str())) != NULL)
    {
        // 打印出所有文件的名字
        while ((ent = readdir(dir)) != NULL)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            {
                continue;
            }
            string temp_name = ent->d_name;
            cout << "文件名:" << temp_name << endl;
            files.push_back(temp_name);
        }
        std::sort(files.begin(), files.end());
        for (int i = 0; i < files.size(); i++)
        {
            string path = dirName;
            string fname = files[i];

            if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
            {

                path.append(fname);
                cout << "文件路径:" << path << endl;
                imageFnames.push_back(path);
            }
            else
            {
                cout << "未找到指定类型的文件" << endl;
            }
        }
        closedir(dir);
    }
}

int main(int argc, char const *argv[])
{
    // 面部检测器
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    // 形状检测器
    dlib::shape_predictor landmarkDetector;

    // 初始化形状检测器
    dlib::deserialize("../common/resources/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    // 读取所有的图片
    string dirName = "../data/images/girls";
    if (!dirName.empty() && dirName.back() != '/')
    {
        dirName += '/';
    }

    // 读取目录下的图片文件
    vector<string> imageNames, ptsNames;
    readFileName(dirName, imageNames);

    // 如果没有文件的话可以退出
    if (imageNames.empty())
    {
        cout << "没有制定类型的文件!" << endl;
        return EXIT_FAILURE;
    }

    // 获取所有图片的脸部关键点
    vector<vector<Point2f>> allPoints;
    vector<Mat> images;
    for (size_t i = 0; i < imageNames.size(); i++)
    {
        Mat img = imread(imageNames[i]);
        if (!img.data)
        {
            cout << "图片:" << imageNames[i] << "读取失败!" << endl;
        }
        else
        {
            vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, img, 1.0);
            if (points.size() > 0)
            {
                allPoints.push_back(points);
                img.convertTo(img, CV_32FC3, 1 / 255.0);
                images.push_back(img);
            }
        }
    }

    if (images.empty())
    {
        cout << "没有检测到有脸的图像" << endl;
        return EXIT_FAILURE;
    }

    int numImages = images.size();

    // 归一化的图片和点
    vector<Mat> imagesNorm;
    vector<vector<Point2f>> pointsNorm;

    // 平局值之后的关键特征点
    vector<Point2f> pointsAvg(allPoints[0].size());

    // 定出输出图像的尺寸
    Size size(600, 600);

    // 定义进行Delaunay剖分的8个边界点
    vector<Point2f> boundaryPts;
    getEightBoundaryPoints(size, boundaryPts);

    for (size_t i = 0; i < images.size(); i++)
    {
        vector<Point2f> points = allPoints[i];

        Mat img_temp;
        normalizeImagesAndLandmarks(size, images[i], img_temp, points, points);

        // 计算标记点的平均位置
        for (size_t k = 0; k < points.size(); k++)
        {
            pointsAvg[k] += points[k] * (1.0 / numImages);
        }

        // 将边界点加入进去
        for (size_t m = 0; m < boundaryPts.size(); m++)
        {
            points.push_back(boundaryPts[m]);
        }

        pointsNorm.push_back(points);
        imagesNorm.push_back(img_temp);
    }

    // 边界点
    for (size_t m = 0; m < boundaryPts.size(); m++)
    {
        pointsAvg.push_back(boundaryPts[m]);
    }

    // Delaunay矩形
    Rect rect(0, 0,size.width, size.height);
    vector<vector<int>> dt;
    calculateDelaunayTriangles(rect, pointsAvg, dt);

    Mat output = Mat::zeros(size, CV_32FC3);

    for(size_t i=0; i<numImages; i++)
    {
        Mat img;
        warpImage(imagesNorm[i], img, pointsNorm[i], pointsAvg, dt);

        output = output + img;

        ostringstream win;
        win << "Result"<< i;

        imshow(win.str(), output);
    }

    output = output / (double)numImages;

    // 将合成图像保存下来
    imwrite("../data/images/result.jpg", output*255);

    imshow("Result", output);
    waitKey(0);

    return 0;
}
