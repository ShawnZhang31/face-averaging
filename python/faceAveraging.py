import os
import sys
import cv2
import dlib
import numpy as np
import faceBlendCommon as fbc

# 读取指定的文件夹下的所有的jpg图片
def readImagesPath(path):
    """
    从指定的文件夹下面读取所有的jpg图片
    参数:
    ============
    path:文件夹路径
    """
    pass

    imagePaths=[]
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:
            print(filePath)
            imagePaths.append(os.path.join(path, filePath))
    
    return imagePaths


# 入口
if __name__ == "__main__":
    # model
    PREDICTOR_PATH = "../common/resources/shape_predictor_68_face_landmarks.dat"

    # face detector
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    dirName = "../data/images/girls"

    # 读取所有的图片
    imagePaths = readImagesPath(dirName)

    if len(imagePaths) == 0:
        print("没有读取到对应的图片")
        sys.exit(0)
    
    images = []
    allPoints = []

    for imagePath in imagePaths:
        im = cv2.imread(imagePath)
        if im is None:
            print("image:{} 读取失败".format(imagePath))
        else:
            print("image:{} 读取成功".format(imagePath))
            points=fbc.getLandmarks(faceDetector, landmarkDetector, im, 2)
            if len(points) > 0:
                allPoints.append(points)
                im = np.float32(im)/255.0
                images.append(im)
            else:
                print("未检测到landmarks")
    
    if len(images) == 0:
        print("没有检测到合适的图像")
        sys.exit(0)
    
    # 定义输出图像的尺寸
    w = 300
    h = 300

    boundaryPts = fbc.getEightBoundaryPoints(h, w)
    
    numImages = len(imagePaths)
    numLandmarks = len(allPoints[0])

    # 归一化图片和特征点
    imagesNorm = []
    pointsNorm = []

    # 平均值
    pointsAvg = np.zeros((numLandmarks,2), dtype=np.float32)

    # 将图片规整到输出坐标系中并求解平均值
    for i, img in enumerate(images):
        points = allPoints[i]

        points = np.array(points)

        # print(points)

        img, points = fbc.normalizeImagesAndLandmarks((h,w),img,points)

        cv2.imshow("{}".format(i), img)

        pointsAvg = pointsAvg + (points/(1.0*numImages))

        # 添加边界点
        points = np.concatenate((points, boundaryPts), axis=0)

        pointsNorm.append(points)
        imagesNorm.append(img)

    pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)
    
    # 计算delaunay三角形
    rect = (0, 0, w, h)
    dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)
    
    # 输出图像
    output = np.zeros((h, w, 3), dtype=np.float)
    
    for i in range(0, numImages):
        imWarp = fbc.warpImage(imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)

        output = output + imWarp
    
    output = output/(1.0*numImages)
    cv2.imshow("Result", output)
    cv2.waitKey(0)
    



