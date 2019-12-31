# 使用Delaunay细分发求取平均图像
使用dlib的面部关键点检测和OpenCV的Delaunay细分求图像的平局图像。人物的平均图像是把人像的图像上的像素点进行累加平均获取的一张平均图像。有研究表明人物的平均比一般的人脸看起来更加舒适，一般的小孩子或者婴儿比较喜欢大众脸，有研究表示也是由于大众脸的人的面部图像更加接近于平均脸。下面是一个平均脸的例子，前六张图片是电影《芳华》六位女主：  
<div style="display:inline">
<img src="/data/images/girls/haoshuwen.jpg" width="120"/>
<img src="/data/images/girls/hexiaoping.jpg" width="120"/>
<img src="/data/images/girls/lindingding.jpg" width="120"/>
<img src="/data/images/girls/xiaobalei.jpg" width="120"/>
<img src="/data/images/girls/xiaohuizi.jpg" width="120"/>
<img src="/data/images/girls/zuoma.jpg" width="170"/>
</div>  

下面的这张图是采用上面的六张图求取的平均脸：  
<div style="display:inline">
<img src="/data/images/result.jpg" width="240"/>
</div>  

## 编译说明  
### 1.CPP工程编译说明  
1. 配置cpp目录下的CMakeLists.txt文件中的OpenCV的路径  
```
cmake_minimum_required(VERSION 2.8.12)

PROJECT(faceAveraging)

#############  基本库配置  ############
find_package(OpenCV PATHS ~/Works/libs/opencv-release/cv-412/install/lib/cmake/opencv4/)
# 如果配置了OpenCV的环境变量可以使用下面的语句代替上面的语句
# find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)

include_directories( ${OpenCV_INCLUDE_DIRS})
include(../common/dlib/dlib/cmake)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()

add_example(faceAveraging)
```
然后进入cpp文件夹，创建build文件夹，进入build文件夹使用cmake进行编译  

```
# 进入cpp文件夹
cd cpp
# 创建build文件夹，并进入build文件夹
mkdir build && cd build
# 使用cmake进行编译
cmake ..
cmake --build . --config Release
```
运行编译之后的文件  

```
# 在cpp文件下执行
./build/faceAveraging
```
### 2. python代码执行
进入python文件夹创建python虚拟环境，并安装依赖  

```
# 创建虚拟环境
virtualenv --no-site-packages venv -p python3
# 激活虚拟环境
source venv/bin/activate 
# 安装依赖
pip install -r requirements.txt
```
在虚拟环境下执行faceAveraging.py脚本  

```
python faceAveraging.py
```
## 工程结构说明  

```
├── common
│   ├── dlib
│   │   ├── dlib    # dlib源码工程存放路径，可自行替换不同版本的dlib
│   ├── fonts   #字体文件存放路径
│   └── resources   # 面部关键点检测的预测模型文件存放路径
├── cpp     #cpp工程路径
├── data
│   └── images
│       └── girls   #测试用的图片存放目录
└── python      # python工程目录
```

