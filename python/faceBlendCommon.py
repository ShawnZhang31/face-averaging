import cv2
import dlib
import numpy as np
import math
import random

def getEightBoundaryPoints(h, w):
    """
    根据图像的尺寸获取边界点
    参数:
    ===========
    h:图像的高度
    w:图像的宽度
    返回值
    ===========
    boundrayPts:边界点
    """
    pass

    boundrayPts=[]
    boundrayPts.append((0,0))
    boundrayPts.append((w/2,0))
    boundrayPts.append((w-1,0))
    boundrayPts.append((w-1, h/2))
    boundrayPts.append((w-1, h-1))
    boundrayPts.append((w/2, h-1))
    boundrayPts.append((0, h-1))
    boundrayPts.append((0, h/2))

    return np.array(boundrayPts, dtype=np.float)

def dlibLandmarksToPoints(shape):
    """
    从landmark中获取关键点的坐标
    参数:
    ==========
    shape:dlib检测到的形状
    返回值:
    ==========
    points:[],返回关键特征点的坐标
    """
    pass

    points=[]
    for p in shape.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points


def getLandmarks(faceDetector, landmarkDetector, im, FACE_DOWNSAMPLE_RATION = 1):
    """
    检测脸部的关键点
    参数:
    =============
    faceDetector:dlib.get_frontal_face_detector(),脸部检测器
    landmarkDetector:形状检测器
    im: 需要检测的图片
    FACE_DOWNSAMPLE_RATION:默认值为1，降采样的比例
    返回值:
    =============
    points:[],检测到的面部关键特征点的坐标
    """
    pass
    
    points = []

    imSmall = cv2.resize(im,None,
                            fx=1.0/FACE_DOWNSAMPLE_RATION,
                            fy=1.0/FACE_DOWNSAMPLE_RATION,
                            interpolation=cv2.INTER_LINEAR)
    
    faceRects = faceDetector(imSmall, 0)
    
    if len(faceRects) > 0:
        maxArea = 0
        maxRect = None
        for face in faceRects:
            if face.area() > maxArea:
                maxArea = face.area()
                maxRect =[face.left(),
                            face.top(),
                            face.right(),
                            face.bottom()]
        
        rect = dlib.rectangle(*maxRect)
        scaledRect = dlib.rectangle(int(rect.left()*FACE_DOWNSAMPLE_RATION),
                                    int(rect.top()*FACE_DOWNSAMPLE_RATION),
                                    int(rect.right()*FACE_DOWNSAMPLE_RATION),
                                    int(rect.bottom()*FACE_DOWNSAMPLE_RATION))
    
        landmarks = landmarkDetector(im, scaledRect)
        points = dlibLandmarksToPoints(landmarks)
    
    return points

            
def normalizeImagesAndLandmarks(outSize, imgIn, points):
    """
    对图像和特征点坐标进行规范化
    参数:
    =============
    outSize:()输出图像的尺寸
    img:归一化操作的图片
    points:归一化操作的点
    返回值:
    =============
    imgNorm:归一化后的图片
    pointsNorm:归一化的点
    """
    pass

    h, w = outSize
    # 输入图像的外眼角点
    eyecornerSrc = [points[36], points[45]]
    # 输出图像的外眼角点
    eyecornerDst = [(np.int(0.3*w),np.int(h/3)),
                    (np.int(0.7*w),np.int(h/3))]

    tform = similarityTransform(eyecornerSrc,eyecornerDst)
    print(tform)

    imgNorm = np.zeros(imgIn.shape, dtype=imgIn.dtype)

    imgNorm = cv2.warpAffine(imgIn, tform, (w, h))

    # 将原始的point 从x2, 改为 x1 x2
    points2 = np.reshape(points, (points.shape[0],1,points.shape[1]))
    
    pointsOut = cv2.transform(points2, tform)

    pointsOut = np.reshape(pointsOut, (points.shape[0], points.shape[1]))

    return imgNorm, pointsOut


def similarityTransform(inPoints, outPoints):
    """
    求解相似矩阵
    参数:
    ==========
    inPoints:输入点
    outPoints:输出点
    返回值
    ==========
    tform:相似矩阵
    """
    pass

    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    inPts= np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin= c60*(inPts[1][0]-inPts[0][0])+s60*(inPts[1][1]-inPts[0][1])+inPts[0][0]
    yin = c60*(inPts[1][1]-inPts[0][1])-s60*(inPts[1][0]-inPts[0][0])+inPts[0][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout= c60*(outPts[1][0]-outPts[0][0])+s60*(outPts[1][1]-outPts[0][1])+outPts[0][0]
    yout = c60*(outPts[1][1]-outPts[0][1])-s60*(outPts[1][0]-outPts[0][0])+outPts[0][1]

    outPts.append([np.int(xout), np.int(yout)])

    if cv2.__version__ > '3.5':
        print('opencv 版本过高,使用estimateAffinePartial2D方法代替estimateRigidTransform')
        _tform = cv2.estimateAffinePartial2D(np.array(inPts), np.array(outPts))
        tform = _tform[0]
    else:
        tform = cv2.estimateRigidTransform(np.array(inPts), np.array(outPts), False)
    # tform = cv2.estimateAffinePartial2D(np.array(inPts),np.array(outPts),False)

    return tform

def calculateDelaunayTriangles(rect, points):
    """
    计算Delaunay三角形
    参数:
    ===========
    rect:分割图形的包围框
    points:分割点
    返回值:
    ===========
    delaunayTri:三角形列表
    """
    pass

    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))
    
    triangleList = subdiv.getTriangleList()

    # 三角形列表
    delaunayTri = []

    for t in triangleList:
        pt =[]
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0,3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0])<1.0 and abs(pt[j][1] - points[k][1])<1.0):
                        ind.append(k)
            if len(ind)==3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    
    return delaunayTri


def rectContains(rect, point):
    """
    检查点是否在矩形框内
    参数:
    ===============
    rect:矩形框
    point:点
    返回值：
    ===============
    True:在框内
    False:不在框内
    """
    pass

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    else:
        return True


def warpImage(imIn, pointsIn, pointsOut, delaunayTri):
    """
    变换图像
    参数:
    ===========
    imIn:输出图像
    pointsIn:输入点
    pointsOut:输出点:
    delaunayTri:三角形
    返回值:
    ============
    imgOut:变形之后的图像
    """
    pass

    h, w, ch = imIn.shape
    imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

    for j in range(0, len(delaunayTri)):
        tin = []
        tout = []

        for k in range(0, 3):
            pIn = pointsIn[delaunayTri[j][k]]
            pIn = constrainPoint(pIn, w, h)

            pOut = pointsOut[delaunayTri[j][k]]
            pOut = constrainPoint(pOut, w, h)

            tin.append(pIn)
            tout.append(pOut)
        
        warpTriangle(imIn, imOut, tin, tout)

    return imOut

def constrainPoint(p, w, h):
    """
    将点约束在矩形框中
    参数:
    ===========
    p:点
    w:宽度
    h:高度
    返回值:
    ===========
    p:约束之后的点
    """
    pass

    p=(min(max(p[0], 0), w-1), min(max(p[1], 0), h-1))
    return p

def warpTriangle(img1, img2, t1, t2):
    """
    变换三角形并进行alpha混合
    参数：
    ==============
    img1:输入图像
    img2:混合后的图像
    t1:输入三角形
    t2:输出三角形
    返回值:
    ==============
    """
    pass

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0,3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0,1.0,1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)


    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def applyAffineTransform(src, srcTri, dstTri, size):
    """
    对图像的三角应用变换
    参数:
    ===========
    src:原始图像
    SRCTri：原始图像的三角形
    dstTri:目标图像的三角形
    size:变换后的三角形的包围框尺寸
    返回值：
    ==============
    dst:变换后的图像
    """
    pass

    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst