#include<stdio.h>
#include "face_recognization.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/generic_image.h"

int main(int argc, char* argv[])
{
    const char* imagepath = argv[2];
    dlib::array2d<dlib::rgb_pixel> m;
    load_image(m, imagepath);
/*    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
*/
    FILE *stream = NULL; 
    stream = fopen("face.data", "wb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 
    
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
    fclose(stream);

    int  id, flag, count;
    DKSSingleDetectionRes box[1];
    box[0].box = {117, 61, 189, 61, 189, 133, 117,133};
    DKSMultiDetectionRes boxes;
    boxes.num = 1;
    boxes.boxes[0] = box[0];
    DKSFaceRegisterParam rgp;
    rgp.index = 0;
    DKSFaceRecognizationParam rcp;
    rcp.index = 0;
    rcp.threshold = 0.6;
    rcp.k = 5;
    
    //注册
    if(*(argv[1]) == '0')
    {
        DKFaceRegisterInit();
        int count = *(argv[2]) - 48;
        for(int i = 0; i < count; i++)
        {
            char*   rgbfilename = argv[2+i+1];
            DKFaceRegisterProcess("face.data", m.nc(), m.nr(), boxes, rgp);
            DKFaceRegisterEnd(count - (i+1) ? 1 : 0, i+1);
        }
    }

    //识别
    if(*(argv[1]) == '1')
    {
        char*   rgbfilename = argv[2];
        DKFaceRecognizationInit();
        id = DKFaceRecognizationProcess("face.data", m.nc(), m.nr(), boxes, rcp);
        DKFaceRecognizationEnd();
        printf("ID:%d\n", id);
    }
    return 0;
}
