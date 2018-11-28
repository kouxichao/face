#  project for hisi3559a.

# 编译成静态库：

```
cd src
make -j$(nproc)
生成libface.a.
```
# 使用说明：

提供六个主要接口函数：

```
// 说明：录入人脸，在学习人脸阶段，获取至少DKMINFACEREGISTERIMGNUM张同一人脸图片
// 初始化，连接sqlite，准备写入人脸特征和语音，初始化学习图片次数为0
void DKFaceRegisterInit();

// 根据检测到的人脸结果计算特征
int DKFaceRegisterProcess(char * rgbfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRegisterParam param);

// 将计算好的特征存入sqlite中（flag为1），或取消学习人脸（flag为0）;count是存入特征的序数。
void DKFaceRegisterEnd(int flag, int count);

// 说明：从已注册的人脸中识别对应的人脸
// 初始化，连接sqlite，获取人脸库中各个人脸的特征
void DKFaceRecognizationInit();

// 运行knn人脸识别，得到识别结果，如果没有识别出或相似度大于某一阈值（在识别参数中定义），则输出null，否则输出识别出的人的index。
int DKFaceRecognizationProcess(char * rgbfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRecognizationParam param);

// 释放人脸识别资源
void DKFaceRecognizationEnd();

```
工具函数:

```
//计算向量内积
float dot(float* fc1, float* fc2);

//规范化向量
int normalize(ncnn::Mat& fc1);

//knn最近邻实现
int knn(std::vector< std::pair<int, float> >& re, int k);
```

结构体：

```
typedef struct
{
    // 顺时针排列
    int x1; // 偏左上角横坐标
    int y1; // 偏左上角纵坐标
    int x2; // 偏右上角横坐标
    int y2; // 偏右上角纵坐标
    int x3; // 偏右下角横坐标
    int y3; // 偏右下角纵坐标
    int x4; // 偏左下角横坐标
    int y4; // 偏左下角纵坐标
}DKSBox;

typedef struct
{
    int tag; // 标签，与宏定义对应
    float confidence; // 置信度
    DKSBox box;
}DKSSingleDetectionRes;

typedef struct
{
    int num; // 当前检测出的物体总数目
    DKSSingleDetectionRes boxes[DKMINFACEREGISTERIMGNUM];
}DKSMultiDetectionRes;

typedef struct
{
    int index; //待识别边界框索引
    float threshold; // 相似度阈值，相似度超过该值认为不能认出该人脸
}DKSFaceRecognizationParam;

typedef struct
{
    int index; //待识别边界框索引
}DKSFaceRegisterParam;
```
# 示例
```
执行命令：
cd src
make MODE="-DJPG_DEMO"
生成可执行文件demo_face,执行需要把models下的文件与其放在同一目录
注:demo_face的输入是jpg或png的文件，而前面生成的库输入是rgb文件。
```


