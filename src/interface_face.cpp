#include <stdio.h>
#include <sqlite3.h>
#include "net.h"
#include "dlib/image_processing/generic_image.h"
#include "dlib/image_io.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "face_recognization.h"

#ifdef JPG_DEMO
//测试使用
static int id= 0;
#endif

static  sqlite3* facefeatures;
static  ncnn::Mat fc;

//工具函数
int dot_int(const int* fc1, const int* fc2)
{
    float sum = 0;
    for(int i=0; i<128; i++)
    {
        sum += fc1[i] * fc2[i];
    }
                      
}
float dot(float* fc1, float* fc2)
{
    float sum = 0;
    for(int i=0; i<128; i++)
    {
        sum += (*(fc1+i)) * (*(fc2+i));
    }
    return sum;
}

int normalize(float* fc1)
{
    float sq = sqrt(dot(fc1, fc1));
    for(int i=0; i<128; i++)
    {
        *(fc1+i) = (*(fc1+i))/sq;
    }
    return 0;
}

int knn(std::vector< std::pair<int, float> >& re, int k, float threshold)
{
    std::pair<int, float> temp;
    for(int i=0; i<re.size(); i++)
    {
	    for(int j=i+1; j<re.size(); j++)
	    {
            if(re[j].second > re[i].second)
            {
            	temp = re[i];
                re[i] = re[j];
                re[j] = temp;
            }
        }
    }
   
    if(re[0].second > threshold)
    { 
        std::vector< std::pair<int, int> > vote;
        vote.push_back(std::make_pair(re[0].first, 1));
        for(int i=1; i<k; i++)
        {
            int j=0;
	        for(; j<vote.size(); j++)
            {
                if(vote[j].first == re[i].first)
                {
                    vote[j].second += 1;
                    break;
                }
            }
            if(j == vote.size())
            {
                vote.push_back(std::make_pair(re[i].first, 1));
            } 
        }
    
        float max = 0;
        // printf("(ID:%d)__ballot:%d\n", vote[0].first, vote[0].second);
        for(int j=1; j<vote.size(); j++)
        {
            if(vote[j].second > vote[0].second)
            {
                max = j;
            }
            // printf("(ID:%d)__ballot:%d\n", vote[j].first, vote[j].second);
        } 
#if DEBUG  
        printf("results:\n(ID:%d)__ballot:%d\n", vote[max].first, vote[max].second);
#endif
        return vote[max].first;
    }
    else
    {
   	return -1; 
    }
}

void DKFaceRegisterInit()
{
    int rc = sqlite3_open("face_feature.db", &facefeatures);
    
    if(rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(facefeatures));
        sqlite3_close(facefeatures);
        exit(1);
    }
    else
    {
        fprintf(stderr, "Opened database successfully\n");
    }
}

int DKFaceRegisterProcess(char* rgbfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRegisterParam param)
{

#ifdef JPG_DEMO
    //使用图片路径进行测试（仅作测试时用,rgbfilename是jpg,png文件路径）
    dlib::array2d<dlib::rgb_pixel> img, face_chips;
    load_image(img, rgbfilename);
#else
    FILE *stream = NULL; 
    stream = fopen(rgbfilename, "rb");   

    if(NULL == stream)
    {
        fprintf(stderr, "error:read imgdata!");
        exit(1);
    }
#if DEBUG
    fprintf(stderr, "width_height: %d, %d", iWidth, iHeight);
#endif
    unsigned char* rgbData = new unsigned char[iHeight*iWidth*3];
    fread(rgbData, 1, iHeight*iWidth*3, stream);
    fclose(stream); 
    
    dlib::array2d<dlib::rgb_pixel> img(iHeight, iWidth), face_chips;

    //（unsigned char*）2（dlib::array2d<dlib::rgb_pixel>）  
    dlib::image_view<dlib::array2d<dlib::rgb_pixel>> imga(img);
/*
// rgb_meta
    for(int r = 0; r < iHeight; r++) 
    {
        const unsigned char* v = rgbData + r * iWidth * 3;
        for(int c = 0; c < iWidth; c++)
        {
            dlib::rgb_pixel p;
            p.red = v[c*3];
            p.green = v[c*3+1];
            p.blue = v[c*3+2];
            assign_pixel( imga[r][c], p );            
        }
    }
*/
// rgb_panel
    const unsigned char* channel_r = rgbData;
    const unsigned char* channel_g = rgbData + iHeight * iWidth;
    const unsigned char* channel_b = rgbData + iHeight * iWidth * 2 ;
    for(int r = 0; r < iHeight; r++) 
    {
        for(int c = 0; c < iWidth; c++)
        {
            dlib::rgb_pixel p;
            p.red = channel_r[r * iWidth + c];
            p.green = channel_g[r * iWidth + c];
            p.blue = channel_b[r * iWidth + c];
            assign_pixel( imga[r][c], p );            
        }
    }

    delete [] rgbData;
#endif  

#ifdef JPG_DEMO
    //脸部对齐
    DKSBox  box = boxes.boxes[id].box;
#else
    DKSBox  box = boxes.boxes[param.index].box;
#endif

    int y_top = box.y1 > box.y2 ? box.y2 : box.y1;
    int y_bottom = box.y3 > box.y4 ? box.y3 : box.y4;
    int x_left = box.x1 > box.x4 ? box.x4 : box.x1;
    int x_right = box.x2 > box.x3 ? box.x2 : box.x3;
    y_top =  y_top > 0 ?  y_top : 0;
    x_left = x_left > 0 ? x_left : 0;
    
#ifdef JPG_DEMO
    y_bottom = y_bottom < img.nr() ? y_bottom : img.nr(); 
    x_right = x_right < img.nc() ? x_right : img.nc(); 
#else
    y_bottom = y_bottom < iHeight ? y_bottom : iHeight; 
    x_right = x_right < iWidth ? x_right : iWidth; 
#endif
  
    dlib::rectangle det(x_left, y_top, x_right, y_bottom);
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    dlib::full_object_detection shape = sp(img, det);
    dlib::extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chips);

    
    int col = (int)face_chips.nc();
    int row = (int)face_chips.nr();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)(&face_chips[0][0]), ncnn::Mat::PIXEL_RGB, col, row, 112, 112);

    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("mobilefacenet.param");
    mobilefacenet.load_model("mobilefacenet.bin");
    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ex.extract("fc1", fc);
    normalize(fc);
}

void DKFaceRegisterEnd(int flag, int count)
{
    //查表插入
    char* zErrMsg;
    const char* sql;
    int registernum = 0;

    sql = "CREATE TABLE IF NOT EXISTS FEATURES("  \
         "NUMFEA         INT     NOT NULL," \
         "FEAOFFACE      BLOB    NOT NULL );";
    int  rc;
    rc = sqlite3_exec(facefeatures, sql, NULL, 0, &zErrMsg);
    if( rc != SQLITE_OK ){
       fprintf(stderr, "SQL error: %s\n", zErrMsg);
       sqlite3_free(zErrMsg);
    }else{
       fprintf(stdout, "Operate on Table FEATURES\n");
    }

    sqlite3_stmt* stat;
    
    if(count > 1 && count <= 10)
    {
        sqlite3_blob* blob = NULL;
        
        //获取行数
        sqlite3_stmt* stat;
        int rc = sqlite3_prepare_v2(facefeatures, "SELECT max(rowid),NUMFEA FROM FEATURES", -1, &stat, NULL);
        if(rc!=SQLITE_OK) {
            fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(facefeatures));
            exit(1);
        }      

        int rid, numfea;
        if( sqlite3_step(stat) == SQLITE_ROW ){
            rid = sqlite3_column_int(stat, 0);
            numfea = sqlite3_column_int(stat, 1);
        } 
        sqlite3_finalize(stat);

        //
        rc = sqlite3_blob_open(facefeatures, "main", "FEATURES", "FEAOFFACE", rid, 1, &blob);
        if (rc != SQLITE_OK)
        {
            printf("Failed to open BLOB: %s \n", sqlite3_errmsg(facefeatures));             
            return ;
        }

        sqlite3_blob_write(blob, fc, 512, numfea*512);
        if (rc != SQLITE_OK)
        {
            fprintf(stderr, "failed to write feature_BLOB!\n");
            return ;
        }
               
        sqlite3_blob_close(blob);
        rc = sqlite3_prepare_v2(facefeatures, " UPDATE FEATURES SET (NUMFEA) = (?) WHERE (rowid) = (?)", -1, &stat, NULL);
        sqlite3_bind_int(stat, 1, numfea+1);
        sqlite3_bind_int(stat, 2, rid);
        rc = sqlite3_step(stat);
        if( rc != SQLITE_DONE ){
        printf("%s",sqlite3_errmsg(facefeatures));
            return ;
        }
        sqlite3_finalize(stat);
        
        registernum = numfea + 1;  
        fprintf(stderr, "Records (rowid)%d insert %dth feature successfully!\n", rid, numfea+1);
    } 
    else
    {   
        float fe[1280] = {0.f};
        for(int j=0; j<128; j++)
        { 
            fe[j] = *((float*)fc.data + j);
        }

        sqlite3_prepare_v2(facefeatures,"INSERT INTO FEATURES (NUMFEA,FEAOFFACE) VALUES(?,?);",-1,&stat, NULL);
        sqlite3_bind_int(stat,1,1);
        sqlite3_bind_blob(stat, 2, fe, 5120, SQLITE_STATIC);
        rc = sqlite3_step(stat);
        if( rc != SQLITE_DONE ){
            printf("%s",sqlite3_errmsg(facefeatures));
            exit(1);
        }
        else{
            fprintf(stderr, "Records created successfully!\n");
        }
        sqlite3_finalize(stat);
    }

    if(flag == 0) 
    {
	    if(registernum >= DKMINFACEREGISTERIMGNUM)
            sqlite3_close(facefeatures); 
        else
        {
            sqlite3_close(facefeatures); 
            fprintf(stderr, "One person needs to register at least %d facial images\n", DKMINFACEREGISTERIMGNUM);
        }
    }

}

void DKFaceRecognizationInit()
{
    //打开数据库
    int rc = sqlite3_open("face_feature.db", &facefeatures);    
    if(rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(facefeatures));
        exit(0);
    }
    else
    {
        fprintf(stderr, "Opened database successfully\n");
    }

}

int DKFaceRecognizationProcess(char* rgbfilename, int iWidth, int iHeight, DKSMultiDetectionRes boxes, DKSFaceRecognizationParam param)
{
#ifdef JPG_DEMO
    //使用图片路径进行测试（仅作测试时用,rgbfilename是jpg,png文件路径）
    dlib::array2d<dlib::rgb_pixel> img, face_chips;
    load_image(img, rgbfilename);
#else    
    FILE *stream = NULL; 
    stream = fopen(rgbfilename, "rb");   

    if(NULL == stream)
    {
        fprintf(stderr, "error:read imgdata!");
        exit(1);
    }

    unsigned char* rgbData = new unsigned char[iHeight*iWidth*3];
    fread(rgbData, 1, iHeight*iWidth*3, stream);
    fclose(stream); 
    
    dlib::array2d<dlib::rgb_pixel> img(iHeight, iWidth), face_chips;

    //（unsigned char*）2（dlib::array2d<dlib::rgb_pixel>）  
    dlib::image_view<dlib::array2d<dlib::rgb_pixel>> imga(img);

    #ifdef RGB_META
    for(int r = 0; r < iHeight; r++) 
    {
        const unsigned char* v = rgbData + r * iWidth * 3;
        for(int c = 0; c < iWidth; c++)
        {
            dlib::rgb_pixel p;
            p.red = v[c*3];
            p.green = v[c*3+1];
            p.blue = v[c*3+2];
            assign_pixel( imga[r][c], p );            
        }
    }

    #else
// rgb_panel
    const unsigned char* channel_r = rgbData;
    const unsigned char* channel_g = rgbData + iHeight * iWidth;
    const unsigned char* channel_b = rgbData + iHeight * iWidth * 2 ;
    for(int r = 0; r < iHeight; r++) 
    {
        for(int c = 0; c < iWidth; c++)
        {
            dlib::rgb_pixel p;
            p.red = channel_r[r * iWidth + c];
            p.green = channel_g[r * iWidth + c];
            p.blue = channel_b[r * iWidth + c];
            assign_pixel( imga[r][c], p );            
        }
    }
    #endif

    delete [] rgbData;
#endif

#ifdef JPG_DEMO
    //脸部对齐
    DKSBox  box = boxes.boxes[id].box;
#else
    DKSBox  box = boxes.boxes[param.index].box;
#endif

    int y_top = box.y1 > box.y2 ? box.y2 : box.y1;
    int y_bottom = box.y3 > box.y4 ? box.y3 : box.y4;
    int x_left = box.x1 > box.x4 ? box.x4 : box.x1;
    int x_right = box.x2 > box.x3 ? box.x2 : box.x3;
    y_top =  y_top > 0 ?  y_top : 0;
    x_left = x_left > 0 ? x_left : 0;

#ifdef JPG_DEMO
    y_bottom = y_bottom < img.nr() ? y_bottom : img.nr(); 
    x_right = x_right < img.nc() ? x_right : img.nc(); 
#else
    y_bottom = y_bottom < iHeight ? y_bottom : iHeight; 
    x_right = x_right < iWidth ? x_right : iWidth; 
#endif
    
#if DEBUG   
    clock_t start = clock();
#endif
    dlib::rectangle det(x_left, y_top, x_right, y_bottom);
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    dlib::full_object_detection shape = sp(img, det);
    dlib::extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chips);

#if DEBUG       
    clock_t finsh = clock();
    fprintf(stderr, "face alignment cost %d ms \n", (finsh-start)/1000);
#endif  
    
    int col = (int)face_chips.nc();
    int row = (int)face_chips.nr();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)(&face_chips[0][0]), ncnn::Mat::PIXEL_RGB, col, row, 112, 112);
#if DEBUG   
    start = clock();
#endif
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("mobilefacenet.param");
    mobilefacenet.load_model("mobilefacenet.bin");
    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ex.extract("fc1", fc);
    normalize(fc);
#if DEBUG       
    finsh = clock();
    fprintf(stderr, "ncnn cost %d ms\n", (finsh-start)/1000);
#endif  
     //获取行数
    sqlite3_stmt* stat;
    int rc = sqlite3_prepare_v2(facefeatures, "SELECT max(rowid) FROM FEATURES", -1, &stat, NULL);
    if(rc!=SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(facefeatures));
        exit(1);
    }      

    int rows;
    if( sqlite3_step(stat) == SQLITE_ROW ){
        rows = sqlite3_column_int(stat, 0);
    } 
    sqlite3_finalize(stat);

    //读取数据库中脸部特征数据
#if DEBUG   
    start = clock();
#endif  
    int i=0;
    float similarity;
    std::vector< std::pair<int, float> > results;
    for(; i< rows; i++)
    {
        sqlite3_blob* blob = NULL;
        rc = sqlite3_blob_open(facefeatures,  "main", "FEATURES", "FEAOFFACE", i+1, 0, &blob);
        if (rc != SQLITE_OK)
        {
            printf("Failed to open BLOB: %s \n", sqlite3_errmsg(facefeatures));
            return -1;
        }
        int blob_length = sqlite3_blob_bytes(blob);
#if DEBUG
    	sqlite3_stmt* stat;
    	int rc = sqlite3_prepare_v2(facefeatures, "SELECT NUMFEA FROM FEATURES WHERE (rowid)=(?)", -1, &stat, NULL);
        sqlite3_bind_int(stat,1,i+1);
    	if(rc!=SQLITE_OK) {
            fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(facefeatures));
       	    exit(1);
    	}      

    	int numfea = 0;
    	if( sqlite3_step(stat) == SQLITE_ROW ){
            numfea = sqlite3_column_int(stat, 0);
    	} 
    	sqlite3_finalize(stat);   
        fprintf(stderr, "num_features:%d\n", numfea);
#endif 
        float buf[128] = {0.f};
        int offset = 0;
        while (offset < blob_length)
        {
            int size = blob_length - offset;
            if (size > 128*4) size = 128*4;
            rc = sqlite3_blob_read(blob, buf, size, offset);
            if (rc != SQLITE_OK)
            {
                printf("failed to read BLOB!\n");
                break;
            }
        
            offset += size;
//	    clock_t start = clock();
            similarity = dot((float*)fc.data, buf);
//	    clock_t finsh = clock();
//     	    fprintf(stderr,"dot cost %d ms\n", (finsh - start));
#if DEBUG   
            fprintf(stderr, "%d_similarity:%f\n",i, similarity);
#endif
            results.push_back(std::make_pair(i, similarity));
        }
        sqlite3_blob_close(blob);
    }
    
    int ID;
    ID = knn(results, param.k, param.threshold);
#if DEBUG   
    finsh = clock();
    fprintf(stderr, "knn cost %d ms\n", (finsh - start)/1000);
#endif   
    return ID;

}

void DKFaceRecognizationEnd()
{
    sqlite3_close(facefeatures);
}
