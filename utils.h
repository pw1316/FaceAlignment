#ifndef __LBF_UTILS_H__
#define __LBF_UTILS_H__

#include "struct.hpp"

#define lbf_roundf(f) ((int)((f) + 0.5f))
#define lbf_C_n_2(n) ((n) * ((n) - 1) / 2)
int clampi(int v, int l, int h);
float randf();
int randM(unsigned int M);

template<class T>
float mean(T* data, int len)
{
    float inv = 1.f / (float)len;
    T sum(0);
    for (int i = 0; i < len; ++i) sum += data[i];
    return inv*sum;
}

#endif
