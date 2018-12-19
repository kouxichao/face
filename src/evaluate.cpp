#include<stdio.h>
#include<unistd.h>
#include<iostream>
#include "face_recognization.h"
#include<cstring>
int main(int argc, char* argv[])
{
    char *root_dir = argv[1];
    FILE *fp = fopen(strcat(root_dir, "bbox.xy"), "r");
    
    int  id, flag, count;
    DKSSingleDetectionRes box[1];
//    box[0].box = {0,0,112,0,112,112,0,112};
    DKSMultiDetectionRes boxes;
    boxes.num = 1;
    boxes.boxes[0] = box[0];
    DKSFaceRegisterParam rgp;
    rgp.index = 0;
    DKSFaceRecognizationParam rcp;
    rcp.index = 0;
    rcp.threshold = 0.6;
    char pre_name[50] = {};

    //注册
    if(*(argv[1]) == '0')
    {
        DKFaceRegisterInit();
        for(int i = 0;; i++)
        {
            char name[50];
            char idx[5];
            int right,left,bottom,top;
            if(fscanf(fp, "%s %s %d,%d,%d,%d", name, idx, &right, &left, &bottom, &top) == EOF);
                break;
            if(strstr(name, "test") == NULL)
            {
                std::string rgbfilename = std::string(root_dir) + std::string(name) + \
                 '/' + "support/" + name + "_" + idx ;
                printf("PATH: %s\n", rgbfilename.data()); 
                if(access(rgbfilename + ".jpg", 0) == -1)
                    rgbfilename = rgbfilename + ".jpg";
                else
                    rgbfilename = rgbfilename + ".png";
                box[0].box = {left,top,right,top,right,bottom,left,bottom};
                DKFaceRegisterProcess(rgbfilename, 100, 100, boxes, rgp);//示例中没有用到iWidth,iHeight两个参数。
                if(strcmp(pre_name, name) == 0)
                    DKFaceRegisterEnd(1, 2);
                else
                    DKFaceRegisterEnd(1, 1);
                strcpy(pre_name,  name);
            }
        }
    }

    //识别
    if(*(argv[1]) == '1')
    {
        char*   rgbfilename = argv[2];
        DKFaceRecognizationInit();
        id = DKFaceRecognizationProcess(rgbfilename, 100, 100, boxes, rcp);//示例中没有用到100,100两个参数。
        DKFaceRecognizationEnd();
        printf("ID:%d\n", id);
    }
    return 0;
}
