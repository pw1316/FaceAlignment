#ifndef __LBF_STURCT_H__
#define __LBF_STURCT_H__

#include <assert.h>

template<class T>
struct T2
{
    T2(T _a1, T _a2) :x(_a1), y(_a2) {}
    T2(T _a1) :x(_a1), y(_a1) {}
    T2() :x(0), y(0) {}
    T2& operator+=(const T2& b)
    {
        x += b.x;
        y += b.y;
        return *this;
    }

    T x, y;
};

template<class T>
T2<T> operator+(const T2<T>& a, const T2<T>& b)
{
    return T2<T>(a.x + b.x, a.y + b.y);
}

template<class T>
T2<T> operator-(const T2<T>& a, const T2<T>& b)
{
    return T2<T>(a.x - b.x, a.y - b.y);
}

template<class T>
T2<T> operator*(T c, const T2<T>& b)
{
    return T2<T>(c*b.x, c*b.y);
}

// dot
template<class T>
T operator*(const T2<T>& a, const T2<T>& b)
{
    return (a.x*b.x + a.y*b.y);
}

typedef T2<float> float2;
typedef T2<double> double2;
typedef T2<int> int2;

template <class T>
struct MatrixColumn;

// row major
template <class T>
struct Matrix2D
{
    // row[i]
    T* operator[](int i) { return data + (i*cols); }
    T* operator[](int i) const { return data + (i*cols); }
    // col(i)
    MatrixColumn<T> column(int i);
    Matrix2D(const Matrix2D<T>& b) :data(0)
    {
        *this = b;
    }
    Matrix2D<T>& operator=(const Matrix2D<T>& b)
    {
        free();
        this->cols = b.cols;
        this->rows = b.rows;
        this->data = new T[cols*rows];
        do_free = true;
        memcpy(this->data, b.data, sizeof(T)*rows*cols);
        return *this;
    }
    Matrix2D(T* d, int r, int c) :data(d), cols(c), rows(r), do_free(false) {}
    Matrix2D(int r, int c) :cols(c), rows(r), do_free(true)
    {
        data = new T[cols*rows];
        memset(data, 0, sizeof(T) * cols * rows);
    }
    ~Matrix2D() { this->free(); }
    void free() { if (do_free && data) delete[] data; }

    T* data;
    int cols, rows;
private:
    bool do_free;
};

template <class T>
struct MatrixColumn
{
    MatrixColumn(T* st, Matrix2D<T>* a) :start(st), arr(a) {}
    MatrixColumn& operator=(T* data)
    {
        T* p = start;
        for (int i = 0; i < arr->rows; ++i)
        {
            *p = data[i];
            p += arr->cols;
        }
        return *this;
    }
    MatrixColumn& operator=(const MatrixColumn& rhs)
    {
        assert(arr->rows == rhs.arr->rows);
        T* p = start, *q = rhs.start;
        for (int i = 0; i < arr->rows; ++i)
        {
            *p = *q;
            p += arr->cols;
            q += rhs.arr->cols;
        }
        return *this;
    }
    void add(const T2<T>& t2)
    {
        T* p = start;
        for (int i = 0; i < arr->rows; ++i)
        {
            if (i & 1)
                *p += t2.y;
            else
                *p += t2.x;
            p += arr->cols;
        }
    }
private:
    T* start;
    Matrix2D<T>* arr;
};

template <class T>
inline MatrixColumn<T> Matrix2D<T>::column(int i) { return MatrixColumn<T>(data + i, this); }

typedef Matrix2D<float> Matrix2Df;
typedef Matrix2D<int> Matrix2Di;

#endif