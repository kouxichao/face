#ifndef FACE_RECOGNIZATION_h
#define FACE_RECOGNIZATION_h
//#include "net.h"
#include<vector>
#ifndef DKMAXBOXNUM
#define DKMAXBOXNUM 30 // 最多可检测对象数目
#endif

#ifndef DKMINFACEREGISTERIMGNUM
#define DKMINFACEREGISTERIMGNUM  5 // 人脸注册时最少需采集人脸数目
#endif

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
    DKSSingleDetectionRes boxes[DKMAXBOXNUM];
}DKSMultiDetectionRes;

typedef struct
{
    int index; //边界框索引
    float threshold; // 相似度阈值，相似度超过该值认为不能认出该人脸
    int k;
}DKSFaceRecognizationParam;

typedef struct
{
    int index; //边界框索引
    float threshold; // 相似度阈值，相似度超过该值认为不能认出该人脸
    int flag;        //
}DKSFaceRegisterParam;


//计算向量内积
float dot(float* fc1, float* fc2);

/**
 * 特征向量归一化
 */
int normalize(float* fc1);

/**
 * 计算输入人脸特征的K最近邻
 * 返回识别结果的rowid
 */
int knn(std::vector< std::pair<int, float> >& re, int k);


/**
 * 人脸注册识别相关函数
 * 录入人脸，在学习人脸阶段，获得少于DKMINFACEREGISTERIMGNUM张注册图片，则无法完成注册,只有在大于DKMINFACEREGISTERIMGNUM,且至少有DKMINFACEREGISTERIMGNUM张合格的图片才会注册成功。
 * 录入人脸前调用DKFaceRegisterInit() ，结束调用DKFaceRegisterEnd()。
 */

/**
 * 初始化，连接sqlite，准备写入人脸特征和语音，初始化学习的图片次数为0
 */
void DKFaceRegisterInit();

/**
 * 根据检测到的人脸结果计算特征，并判断在最后一次调用时判断此次注册是否合格。
 *
 * @param imgfilename 待学习图片路径
 * @param iWidth      图片宽
 * @param iHeight     图片高
 * @param boxes       图片中的人脸框
 * @param param       其他学习参数
 * @return 1  学习成功
 *         0  学习进行中
 *         -1 学习失败
 */
int DKFaceRegisterProcess(char * imgfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRegisterParam param);

//参数1：将计算好的特征存入sqlite中（flag为1），或取消学习人脸（flag为0）关闭数据库。参数2：是否为新添加的人（增加记录数）。参数3:添加特征的行id（以1开始）注：当newPerson=true时，忽略参数rid。
void DKFaceRegisterEnd(int flag, bool newPerson, int rid); 

//参数1：将计算好的特征存入sqlite中（flag为1），或取消学习人脸（flag为0）关闭数据库。参数2：要存入的特征序数（为1时表示创建新的记录即有新人注册）。
void DKFaceRegisterEnd(const char* voiceFile);

// 说明：从已注册的人脸中识别对应的人脸
// 初始化，连接sqlite，获取人脸库中各个人脸的特征
void DKFaceRecognizationInit();

int DKFaceRecognizationProcess(char* recvoicefile, const char * imgfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRecognizationParam param);

// 释放人脸识别资源
void DKFaceRecognizationEnd();

#endif 
