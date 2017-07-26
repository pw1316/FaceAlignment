#include "image.h"

using std::string;

Image::~Image()
{
    if (data) delete[] data;
    if (shape) delete[] shape;
}

uint8* Image::operator[](int i)
{
    return data + ((height - 1 - i)*width); // y upside down
}

uint8* Image::operator[](int i) const
{
    return data + ((height - 1 - i)*width); // y upside down
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

        this->data = new uint8[w*h];
        BYTE *bits = (BYTE*)FreeImage_GetBits(dib);
        int src_linew = FreeImage_GetLine(dib);
        for (int y = 0; y < h; ++y)
        {
            BYTE* src = bits + y * src_linew;
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

void Image::plotShape(float* shp)
{
    if (!shape) shape = new int[2 * NUM_LANDMARKS];
    for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
        shape[i] = lbf_roundf(shp[i]);
}

void Image::saveAs(const string& fpath)
{
    const char* filename = fpath.c_str();
    FIBITMAP*dib = FreeImage_ConvertFromRawBits(data, width, height, sizeof(uint8)*width, 8, 0, 0, 0);

    if (shape)
    {
        int val = 255;
        for (int i = 0; i < NUM_LANDMARKS; ++i)
        {
            int _x = shape[i + i], _y = shape[i + i + 1];
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
