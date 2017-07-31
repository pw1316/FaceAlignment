#ifndef __LBF_IMAGE_H__
#define __LBF_IMAGE_H__

#include "lbf.h"
#include "utils.h"
#include "timer.h"
#include "3rdparty/FreeImage.h"
#include <string>

typedef unsigned char uint8;
struct Image
{
    Image() :data(0), shape(0), width(0), height(0) {}
    ~Image();
    uint8* operator[](int i);
    uint8* operator[](int i) const;
    bool load(const std::string& fpath);
    bool detectFace();
    bool detectFaceAndNormalize(float* shp);
    void plotShape(float* shp);
    void saveAs(const std::string& fpath);
    void saveAsNoShape(const std::string& fpath);

    uint8* data;
    int2 faceLeftTop;
    int faceWidth, faceHeight;
    int* shape; // plot shape
    int width, height;
};

#endif

