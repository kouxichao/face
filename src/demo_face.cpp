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
    
#ifndef RGB_META
    unsigned char* rgb_panel = new unsigned char[m.nc()*m.nr()*3];
    unsigned char* rgb_panel_r = rgb_panel;
    unsigned char* rgb_panel_g = rgb_panel + m.nc()*m.nr();
    unsigned char* rgb_panel_b = rgb_panel + m.nc()*m.nr()*2;    

    for(int r = 0; r < m.nr(); r++)
    {
	for(int c = 0; c < m.nc(); c++)
	{
	    rgb_panel_r[r*m.nc()+c] = m[r][c].red;
	    rgb_panel_g[r*m.nc()+c] = m[r][c].green;
	    rgb_panel_b[r*m.nc()+c] = m[r][c].blue;
	}
    }
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fwrite(rgb_panel, 1, m.nc()*m.nr()*3, stream);
#else
    
    fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
#endif
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fclose(stream);

    int  id, flag, count;
    DKSSingleDetectionRes box[1];
    box[0].box = {157, 141, 229, 141, 229, 213, 157, 213};//229,157,213,141 189,117,133,61
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
     //   int count = *(argv[3]) - 48;
        for(int i = 0; i < 10; i++)
        {
       //     char*   rgbfilename = argv[2+i+1];
            DKFaceRegisterProcess("face.data", m.nc(), m.nr(), boxes, rgp);
printf("re\n");
	    DKFaceRegisterEnd(1, i+1);
          //  DKFaceRegisterEnd(count - (i+1) ? 1 : 0, i+1);
        }
    }

    //识别
    if(*(argv[1]) == '1')
    {
        char*   rgbfilename = argv[2];
        DKFaceRecognizationInit();
        for(int i = 0; i < 200; i++)
        {
            id = DKFaceRecognizationProcess("face.data", m.nc(), m.nr(), boxes, rcp);
            printf("%d_ID:%d\n", i,id);
        }
        DKFaceRecognizationEnd();
    }
    return 0;
}
