#include "utils.h"
#include "3rdparty/mt19937/mt19937ar.h"

#include <math.h>
#include <omp.h>

#define min(a, b) (((a) < (b)) ? (a):(b))
#define max(a, b) (((a) > (b)) ? (a):(b))

static omp_lock_t rnglock;

int clampi(int v, int l, int h)
{
    return min(h, max(l, v));
}

static int _srand()
{
    static int seeded = 0;
    if (!seeded)
    {
        seeded = 1;
        unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4;
        init_by_array(init, length);
        omp_init_lock(&rnglock);
    }
    return 1;
}

float randf()
{
    static int __dummy = _srand();
    omp_set_lock(&rnglock);
    float ret = (float)genrand_real2();
    omp_unset_lock(&rnglock);
    return ret;
}

int randM(unsigned int M)
{
    static int __dummy = _srand();
    omp_set_lock(&rnglock);
    int ret = (int)(((unsigned __int64)genrand_int32()*M) >> 32);
    omp_unset_lock(&rnglock);
    return ret;
}
