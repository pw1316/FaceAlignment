#include "image.h"
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;
using namespace cv;

/* Assertion */
#undef assert
static void _pwassert(const wchar_t* expression, const wchar_t* file, unsigned line)
{
    printf("PW Assertion failed: %ws\n", expression);
    printf("%ws(%u)\n", file, line);
    volatile int i = 0;
    i = 1 / i;
}
#define assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_pwassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )

static bool checkShapeInFace(Rect r, float* shp)
{
    float sumX = 0;
    float sumY = 0;
    float maxX = 0, minX = FLT_MAX, maxY = 0, minY = FLT_MAX;
    for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; landmarkId++) {
        if (shp[2 * landmarkId] > maxX) maxX = shp[2 * landmarkId];
        if (shp[2 * landmarkId] < minX) minX = shp[2 * landmarkId];
        if (shp[2 * landmarkId + 1] > maxY) maxY = shp[2 * landmarkId + 1];
        if (shp[2 * landmarkId + 1] < minY) minY = shp[2 * landmarkId + 1];
        sumX += shp[2 * landmarkId];
        sumY += shp[2 * landmarkId + 1];
    }
    if ((maxX - minX) > r.width*1.5) {
        return false;
    }
    if ((maxY - minY) > r.height*1.5) {
        return false;
    }
    if (abs(sumX / NUM_LANDMARKS - (r.x + r.width / 2.0f)) > r.width / 2.0f) {
        return false;
    }
    if (abs(sumY / NUM_LANDMARKS - (r.y + r.height / 2.0f)) > r.height / 2.0f) {
        return false;
    }
    return true;
}

Image::~Image()
{
    if (data) delete[] data;
    if (shape) delete[] shape;
}

uint8* Image::operator[](int i)
{
    return data + ((height - 1 - i) * width); // y upside down
}

uint8* Image::operator[](int i) const
{
    return data + ((height - 1 - i) * width); // y upside down
}

bool Image::load(const string & fpath)
{
    if (data) delete[] data;
    width = 0;
    height = 0;
    const char* filename = fpath.c_str();

    FREE_IMAGE_FORMAT fifmt = FreeImage_GetFileType(filename, 0);
    if (fifmt == FIF_UNKNOWN)
        fifmt = FreeImage_GetFIFFromFilename(filename);
    if (fifmt != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fifmt))
    {
        FIBITMAP* dib = FreeImage_Load(fifmt, filename, 0);
        if (!dib) return false;
        int w = FreeImage_GetWidth(dib);
        int h = FreeImage_GetHeight(dib);
        FIBITMAP* gdib = FreeImage_ConvertToGreyscale(dib);
        FreeImage_Unload(dib);
        dib = gdib;

        this->data = new uint8[w * h];
        BYTE* bits = (BYTE*)FreeImage_GetBits(dib);
        int fiwidth = FreeImage_GetLine(dib);
        for (int y = 0; y < h; ++y)
        {
            BYTE* src = bits + y * fiwidth;
            memcpy(this->data + y * w, src, sizeof(uint8) * w);
        }
        FreeImage_Unload(dib);
        width = w;
        height = h;
        return true;
    }
    else
    {
        return false;
    }
}

bool Image::detectFace()
{
    CascadeClassifier cc;
    if (!cc.load(CASCADE_NAME))
    {
        printf("Can't load cascade\n");
        return false;
    }
    this->saveAsNoShape("~temp.png");
    Mat cvImage = imread("~temp.png", IMREAD_GRAYSCALE);
    vector<Rect> cvFaces;
    Rect r = Rect(0, 0, 0, 0);
    equalizeHist(cvImage, cvImage);
    cc.detectMultiScale(cvImage, cvFaces, 1.1, 2);
    assert(cvFaces.size() > 0);
    r = cvFaces[cvFaces.size() - 1];
    this->faceLeftTop.x = r.x;
    this->faceLeftTop.y = r.y;
    this->faceWidth = r.width;
    this->faceHeight = r.height;
    system("del \"~temp.png\"");
    return true;
}

// Train Only. Assuming 1 face
bool Image::detectFaceAndNormalize(float* shp)
{
    // Detect
    CascadeClassifier cc;
    if (!cc.load(CASCADE_NAME))
    {
        printf("Can't load cascade\n");
        return false;
    }
    this->saveAsNoShape("~temp.png");
    Mat cvImage = imread("~temp.png", IMREAD_GRAYSCALE);
    vector<Rect> cvFaces;
    Rect r = Rect(0, 0, 0, 0);
    equalizeHist(cvImage, cvImage);
    cc.detectMultiScale(cvImage, cvFaces);
    if (cvFaces.size() <= 0)
    {
        r.x = 0;
        r.y = 0;
        r.width = width;
        r.height = height;
    }
    else
    {
        int faceId;
        for (faceId = 0; faceId < cvFaces.size(); ++faceId)
        {
            r = cvFaces[faceId];
            if (checkShapeInFace(r, shp))
                break;
        }
        if (faceId >= cvFaces.size())
        {
            r.x = 0;
            r.y = 0;
            r.width = width;
            r.height = height;
        }
    }
    this->faceLeftTop.x = r.x;
    this->faceLeftTop.y = r.y;
    this->faceWidth = r.width;
    this->faceHeight = r.height;
    system("del \"~temp.png\"");

    // Normalize
    for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
    {
        float& x = shp[2 * landmarkId];
        float& y = shp[2 * landmarkId + 1];
        x = (x - faceLeftTop.x) / faceWidth;
        y = (y - faceLeftTop.y) / faceHeight;
    }
    return true;
}

void Image::plotShape(float* shp)
{
    if (!shape) shape = new int[2 * NUM_LANDMARKS];
    for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
    {
        // y
        if (i & 1)
        {
            shape[i] = lbf_roundf(shp[i] * faceHeight + faceLeftTop.y);
        }
        // x
        else
        {
            shape[i] = lbf_roundf(shp[i] * faceWidth + faceLeftTop.x);
        }
    }
}

void Image::saveAs(const string& fpath)
{
    const char* filename = fpath.c_str();
    FIBITMAP* dib = FreeImage_ConvertFromRawBits(data, width, height, sizeof(uint8)*width, 8, 0, 0, 0);

    if (shape)
    {
        int val = 255;
        for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
        {
            int _x = shape[2 * landmarkId], _y = shape[2 * landmarkId + 1];
            const int sz = 1;
            int sx = clampi(_x - sz, 0, width - 1), ex = clampi(_x + sz, 0, width - 1);
            int sy = clampi(_y - sz, 0, height - 1), ey = clampi(_y + sz, 0, height - 1);

            for (int y = sy; y <= ey; ++y)
            {
                BYTE* imgline = FreeImage_GetScanLine(dib, height - 1 - y);
                for (int x = sx; x <= ex; ++x)
                {
                    imgline[x] = val;
                }
            }
        }
    }
    FreeImage_Save(FIF_PNG, dib, filename);
    FreeImage_Unload(dib);
}

void Image::saveAsNoShape(const std::string & fpath)
{
    const char* filename = fpath.c_str();
    FIBITMAP* dib = FreeImage_ConvertFromRawBits(data, width, height, sizeof(uint8)*width, 8, 0, 0, 0);
    FreeImage_Save(FIF_PNG, dib, filename);
    FreeImage_Unload(dib);
}
