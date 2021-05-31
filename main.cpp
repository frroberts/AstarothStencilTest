
#include <iostream>
#include <chrono>

#include <cuda_runtime_api.h>
#include <random>
#include <iomanip> 

/*
struct int3{
    int x;
    int y;
    int z;
};

struct double3{
    double x;
    double y;
    double z;
};
*/

constexpr int xThreads = 16;
constexpr int yThreads = 8;
constexpr int zThreads = 4;


typedef double AcReal;
typedef double3 AcReal3;

typedef struct {
    AcReal3 row[3];
} AcMatrix;

struct AcRealData
{
    AcReal value;
    AcReal3 gradient;
    AcMatrix hessian;
};


__constant__ int AC_mx;
__constant__ int AC_my;
__constant__ int AC_mz;
__constant__ AcReal AC_inv_dsx;
__constant__ AcReal AC_inv_dsy;
__constant__ AcReal AC_inv_dsz;
__constant__ int3 start;
__constant__ int3 end;

__constant__ int xModLut[256];
__constant__ int yModLut[256];
__constant__ int zModLut[256];

__constant__ int xIndLut[(xThreads+6)*(yThreads+6)*(zThreads+6)];
__constant__ int yIndLut[(xThreads+6)*(yThreads+6)*(zThreads+6)];
__constant__ int zIndLut[(xThreads+6)*(yThreads+6)*(zThreads+6)];


__device__
void storeDouble(int ind, double val, double *ptr, int offset)
{
    uint32_t *ptr4 = reinterpret_cast<uint32_t*>(ptr);

    uint32_t lower = (uint64_t)(val);
    uint32_t upper = (uint64_t)(val)>> 32;

    ptr4[ind] = lower;
    ptr4[ind+offset+1] = upper;
}


__device__
double loadDouble(int ind, const double* ptr, int offset)
{
    const uint32_t *ptr4 = reinterpret_cast<const uint32_t*>(ptr);
    double v2 = (double)(ptr4[ind] | ((uint64_t)(ptr4[ind+offset+1])<<32));
    return v2;
}


static __device__
size_t acVertexBufferIdx(const int i, const int j, const int k)
{
    return i +                          //
           j * AC_mx + //
           k * AC_mx * AC_my;
}

static __device__
size_t acVertexBufferIdx_shared(const int i, const int j, const int k)
{
    return i +                          //
           j * (xThreads+6) + //
           k * (xThreads+6) * (yThreads+6);
}


static __device__ int
IDX(const int i, const int j, const int k)
{
    return acVertexBufferIdx(i, j, k);
}

static __device__ int
IDX_shared(const int i, const int j, const int k)
{
    return acVertexBufferIdx_shared(i, j, k);
}

static __device__ constexpr
int IDX(const int i)
{
    return i;
}

static __device__ __forceinline__
int IDX(const int3 idx)
{
    return IDX(idx.x, idx.y, idx.z);
}

static __device__ __forceinline__
AcReal getData(int3 vertexIdx, int3 vertexOffsets, const AcReal *__restrict__ arr)
{
    int3 vertexIdxReal = vertexIdx;

#ifdef SHAREDCACHE
    // if shared 
    vertexIdxReal.x = threadIdx.x + 3;
    vertexIdxReal.y = threadIdx.y + 3;
    vertexIdxReal.z = threadIdx.z + 3;
#ifdef STORE32
    AcReal ret = loadDouble(IDX_shared(vertexIdxReal.x + vertexOffsets.x, vertexIdxReal.y + vertexOffsets.y, vertexIdxReal.z + vertexOffsets.z), arr, (xThreads+6)*(yThreads+6)*(zThreads+6));
#else    
    AcReal ret = arr[IDX_shared(vertexIdxReal.x + vertexOffsets.x, vertexIdxReal.y + vertexOffsets.y, vertexIdxReal.z + vertexOffsets.z)];
#endif
    return ret;

#else
    // if not shared
    return arr[IDX(vertexIdxReal.x + vertexOffsets.x, vertexIdxReal.y + vertexOffsets.y, vertexIdxReal.z + vertexOffsets.z)];
#endif
}


// derivates

static __device__ __forceinline__
AcReal first_derivative(AcReal pencil [ ], AcReal inv_ds){
    AcReal coefficients[] = {0, AcReal(3.0) / AcReal(4.0), -AcReal(3.0) / AcReal(20.0), AcReal(1.0) / AcReal(60.0)};
    AcReal res = 0;
    for(int i = 1; i <= ((6 )/2 ); ++ i ){
        res += coefficients [IDX(i )]*(pencil [IDX(((6 )/2 )+i )]-pencil [IDX(((6 )/2 )-i )]);
    }
    return res *inv_ds;
}

static __device__ __forceinline__
AcReal second_derivative(AcReal pencil [ ], AcReal inv_ds){
    AcReal coefficients[] = {-AcReal(49.0) / AcReal(18.0), AcReal(3.0) / AcReal(2.0), -AcReal(3.0) / AcReal(20.0), AcReal(1.0) / AcReal(90.0)};
    AcReal res = coefficients[IDX(0)] * pencil[IDX(((6) / 2))];
    for(int i = 1; i <= ((6 )/2 ); ++ i ){
        res += coefficients [IDX(i )]*(pencil [IDX(((6 )/2 )+i )]+pencil [IDX(((6 )/2 )-i )]);
    }
    return res *inv_ds *inv_ds;
}

static __device__ __forceinline__
AcReal cross_derivative(AcReal pencil_a [ ], AcReal pencil_b[], AcReal inv_ds_a, AcReal inv_ds_b){
    AcReal fac = AcReal(1.0) / AcReal(720.0);
    AcReal coefficients[] = {AcReal(0.0) * fac, AcReal(270.0) * fac, -AcReal(27.0) * fac, AcReal(2.0) * fac};
    AcReal res = AcReal(0.0);
    for(int i = 1; i <= ((6 )/2 ); ++ i ){
        res += coefficients [IDX(i )]*(pencil_a [IDX(((6 )/2 )+i )]+pencil_a [IDX(((6 )/2 )-i )]-pencil_b [IDX(((6 )/2 )+i )]-pencil_b [IDX(((6 )/2 )-i )]);
    }
    return res *inv_ds_a * inv_ds_b;
}

// dre functions

static __device__ __forceinline__
AcReal derx(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,0}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z)];
    }
    return first_derivative(pencil, AC_inv_dsx);
}

static __device__ __forceinline__
AcReal derxx(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,0}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z)];
    }
    return second_derivative(pencil, AC_inv_dsx);
}

static __device__ __forceinline__
AcReal derxy(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,offset - (6) / 2,0}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,(6) / 2 - offset,0}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + (6) / 2 - offset, vertexIdx.z)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal derxz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,offset - (6) / 2}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,(6) / 2 - offset}, arr);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + (6) / 2 - offset)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal dery(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {0,(6) / 2 - offset,0}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    return first_derivative(pencil, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal deryy(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {0,(6) / 2 - offset,0}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    return second_derivative(pencil, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal deryz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = getData(vertexIdx, {0,offset - (6) / 2,offset - (6) / 2}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + offset - (6) / 2)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = getData(vertexIdx, {0,offset - (6) / 2,(6) / 2 - offset}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + (6) / 2 - offset)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsy, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal derz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {0,0,offset - (6) / 2}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    return first_derivative(pencil, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal derzz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = getData(vertexIdx, {0,0,offset - (6) / 2}, arr);//arr[IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    return second_derivative(pencil, AC_inv_dsz);
}


// helps non share cached, not really when caching in shared memory
// access functions
static __device__ __forceinline__
void preprocessed_data(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex, AcRealData &data) {
    data.value = getData(vertexIdx, {0,0,0}, vertex);//vertex[IDX(vertexIdx)];

    AcReal pencil[(6) + 1];
    pencil[3] = data.value;
    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,0}, vertex);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z)];
    }
    data.gradient.x = first_derivative(pencil, AC_inv_dsx);//derx(vertexIdx, vertex);
    data.hessian.row[IDX(0)].x = second_derivative(pencil, AC_inv_dsx);//derxx(vertexIdx, vertex);


    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {0,(6) / 2 - offset,0}, vertex);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    data.gradient.y = first_derivative(pencil, AC_inv_dsy);
    data.hessian.row[IDX(1)].y = second_derivative(pencil, AC_inv_dsy);


    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {0,0,offset - (6) / 2}, vertex);//arr[IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    data.gradient.z = first_derivative(pencil, AC_inv_dsz);
    data.hessian.row[IDX(2)].z = second_derivative(pencil, AC_inv_dsz);


    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,offset - (6) / 2,0}, vertex);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    AcReal pencil_b[(6) + 1];
    pencil_b[3] = data.value;
    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil_b[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,(6) / 2 - offset,0}, vertex);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + (6) / 2 - offset, vertexIdx.z)];
    }
    data.hessian.row[IDX(0)].y =  cross_derivative(pencil, pencil_b, AC_inv_dsx, AC_inv_dsy);//derxy(vertexIdx, vertex);
    data.hessian.row[IDX(1)].x = data.hessian.row[IDX(0)].y;

    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,offset - (6) / 2}, vertex);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil_b[IDX(offset)] = getData(vertexIdx, {offset - (6) / 2,0,(6) / 2 - offset}, vertex);//arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + (6) / 2 - offset)];
    }
    data.hessian.row[IDX(0)].z = cross_derivative(pencil, pencil_b, AC_inv_dsx, AC_inv_dsz);;//derxz(vertexIdx, vertex);
    data.hessian.row[IDX(2)].x = data.hessian.row[IDX(0)].z;


    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil[IDX(offset)] = getData(vertexIdx, {0,offset - (6) / 2,offset - (6) / 2}, vertex);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + offset - (6) / 2)];
    }
    for (int offset = 0; offset < (6) + 1; ++offset) {
        if(offset == 3)
            continue;
        pencil_b[IDX(offset)] = getData(vertexIdx, {0,offset - (6) / 2,(6) / 2 - offset}, vertex);//arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + (6) / 2 - offset)];
    }
    data.hessian.row[IDX(1)].z = cross_derivative(pencil, pencil_b, AC_inv_dsy, AC_inv_dsz);;//deryz(vertexIdx, vertex);
    data.hessian.row[IDX(2)].y = data.hessian.row[IDX(1)].z;

    return;
}

static __device__ __forceinline__
AcReal preprocessed_value(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex) {
    return getData(vertexIdx, {0,0,0}, vertex);//vertex[IDX(vertexIdx)];
}


static __device__ __forceinline__
AcReal3 preprocessed_gradient(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex) {
    return (AcReal3) {derx(vertexIdx, vertex), dery(vertexIdx, vertex), derz(vertexIdx, vertex)};
}

static __device__ __forceinline__
AcMatrix preprocessed_hessian(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex) {
    AcMatrix mat;
    mat.row[IDX(0)] = (AcReal3) {derxx(vertexIdx, vertex), derxy(vertexIdx, vertex), derxz(vertexIdx, vertex) };
    mat.row[IDX(1)] = (AcReal3) {mat.row[IDX(0)].y, deryy(vertexIdx, vertex), deryz(vertexIdx, vertex) };
    mat.row[IDX(2)] = (AcReal3) {mat.row[IDX(0)].z, mat.row[IDX(1)].z, derzz(vertexIdx, vertex) };
    return mat;
}



// read function

static __device__ __forceinline__
AcRealData read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf, AcReal *__restrict__ sharedBuf) {
    AcRealData data;

#ifdef SHAREDCACHE

    int idxLocal = threadIdx.x + (threadIdx.y * xThreads) + (threadIdx.z * xThreads * yThreads);
#ifdef FLATFILL

    for (size_t i = idxLocal; i < (xThreads+6) * (yThreads+6) * (zThreads+6); i += xThreads * yThreads * zThreads)
    {
#ifdef MODLUT
#pragma message "MODLUT"
        int x = xModLut[i&0xff];
        int xDiv = (i / (xThreads+6));
        int y = yModLut[xDiv&0xff];
        int z = (xDiv/(yThreads+6));
#elif FLOATIND
#pragma message "FLOATIND"
        int xDiv = ((float)i / (float)(xThreads+6));
        int yDiv = ((float)xDiv/(float)(yThreads+6));
        int x = i - (xDiv * (xThreads+6));
        int y = xDiv - ((yDiv)*(yThreads+6));
        int z = yDiv;
#elif INDLUT
#pragma message "INDLUT"
        int x = xIndLut[i];
        int y = yIndLut[i];
        int z = zIndLut[i];
#else
#pragma message "NORMAL"
        int xDiv = (i / (xThreads+6));
        int x = i % (xThreads+6);
        int y = (xDiv)%(yThreads+6);
        int z = (xDiv/(yThreads+6));
/*
        int yDiv = (xDiv/(yThreads+6));
        int x = i - (xDiv * (xThreads+6));
        int y = xDiv - ((yDiv)*(yThreads+6));
        int z = yDiv;
        */
#endif

        //int sharedInd = x + (y * (xThreads+6)) + (z *(xThreads+6)*(yThreads+6));
        int targetX = (blockIdx.x * blockDim.x) + x;
        int targetY = (blockIdx.y * blockDim.y) + y;
        int targetZ = (blockIdx.z * blockDim.z) + z;
        
        if(targetX >= AC_mx || targetY >= AC_my || targetZ >= AC_mz)
            continue;
#ifdef STORE32
        storeDouble(i, buf[IDX(targetX, targetY, targetZ)], sharedBuf, (xThreads+6)*(yThreads+6)*(zThreads+6));
#else
        sharedBuf[i] = buf[IDX(targetX, targetY, targetZ)];
#endif
    }
#elif HYBRIDFILL

#else
    // yes these are ordered wrong ... but faster this way?
    for (size_t x = threadIdx.x; x < xThreads+6; x += xThreads)
    {
        for (size_t y = threadIdx.y; y < yThreads+6; y += yThreads)
        {
            for (size_t z = threadIdx.z; z < zThreads+6; z += zThreads)
            {
                int sharedInd = x + (y * (xThreads+6)) + (z *(xThreads+6)*(yThreads+6));
                int targetX = (blockIdx.x * blockDim.x) + x;
                int targetY = (blockIdx.y * blockDim.y) + y;
                int targetZ = (blockIdx.z * blockDim.z) + z;
                if(targetX >= AC_mx || targetY >= AC_my || targetZ >= AC_mz)
                    continue;
#ifdef STORE32
                storeDouble(sharedInd, buf[IDX(targetX, targetY, targetZ)], sharedBuf, (xThreads+6)*(yThreads+6)*(zThreads+6));
#else
                sharedBuf[sharedInd] = buf[IDX(targetX, targetY, targetZ)];
#endif
            }
        }
    }
#endif
#endif

/*
   for (size_t i = 0; i < (xThreads+6)*(yThreads+6)*(zThreads+6); i++)
   {
       if(sharedBuf[i] != 1 && idxLocal==0)
       {
           printf("fail %i %d \n", i, sharedBuf[i]);
           break;
       }
   }
  */ 
    
    __syncthreads();
    
    if (!(vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z))
    {
#ifdef SHAREDCACHE
#ifdef PROCALLINONE
        preprocessed_data(vertexIdx, globalVertexIdx, sharedBuf, data);
#else
        data.value = preprocessed_value(vertexIdx, globalVertexIdx, sharedBuf);
        data.gradient = preprocessed_gradient(vertexIdx, globalVertexIdx, sharedBuf);
        data.hessian = preprocessed_hessian(vertexIdx, globalVertexIdx, sharedBuf);
#endif
#else
#ifdef PROCALLINONE
        preprocessed_data(vertexIdx, globalVertexIdx, buf, data);
#else
        data.value = preprocessed_value(vertexIdx, globalVertexIdx, buf);
        data.gradient = preprocessed_gradient(vertexIdx, globalVertexIdx, buf);
        data.hessian = preprocessed_hessian(vertexIdx, globalVertexIdx, buf);
#endif
#endif
    }
    return data;
}


// kernel that does the call

__global__ void kern(AcReal* __restrict__ buf, AcReal* __restrict__ bufOut){
    //
    const int3 vertexIdx       = (int3){threadIdx.x + blockIdx.x * blockDim.x + start.x,
                                  threadIdx.y + blockIdx.y * blockDim.y + start.y,
                                  threadIdx.z + blockIdx.z * blockDim.z + start.z};

    #ifdef SHAREDCACHE
    __shared__ AcReal sharedBuf[(xThreads+6)*(yThreads+6)*(zThreads+6)+1];
    #else
    AcReal *sharedBuf; // leave uninitialized 
    #endif

    // we cant exit the threads since i need them to fill the shared buffer
    //if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
    //    return;

    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

    AcRealData dat = read_data(vertexIdx, vertexIdx, buf, sharedBuf);

    if (!(vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z))
    {

// sum it all together to stop it being optimized
    bufOut[idx] = dat.value + dat.gradient.x + dat.gradient.y + dat.gradient.z + 
        dat.hessian.row[0].x + dat.hessian.row[0].y + dat.hessian.row[0].z + 
        dat.hessian.row[1].x + dat.hessian.row[1].y + dat.hessian.row[1].z + 
        dat.hessian.row[2].x + dat.hessian.row[2].y + dat.hessian.row[2].z;
    }
}

__global__ void filler(AcReal* __restrict__ buf, AcReal* __restrict__ bufOut, int count)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    if(idx >= count)
        return;
    buf[idx] = 1;// + threadIdx.x/120.0;
    bufOut[idx] = 0;
}



int main(int argc, char const *argv[]) {
    //

    int h_AC_mx = std::atoi(argv[1]);
    int h_AC_my = std::atoi(argv[2]);
    int h_AC_mz = std::atoi(argv[3]);
    double h_AC_inv_dsx = 1;
    double h_AC_inv_dsy = 1;
    double h_AC_inv_dsz = 1;
    int3 h_start;
    int3 h_end;

    int count = h_AC_mx * h_AC_my * h_AC_mz;

    AcReal *inBuf;
    AcReal *outBuf;
    AcReal *hostBuf = new AcReal[count];
    AcReal *hostIn = new AcReal[count];


    std::mt19937 gen(2);
    std::uniform_real_distribution<> dis(1.0, 2.0);
    for (int i = 0; i < count; ++i) {
        hostIn[i] = dis(gen);
    }

    cudaMalloc((void**)&inBuf, count * sizeof(AcReal));
    cudaMalloc((void**)&outBuf, count * sizeof(AcReal));

    filler<<<1+(count/512), 512>>>(inBuf, outBuf, count);
    
    cudaMemcpy(inBuf, hostIn, sizeof(AcReal)*count, cudaMemcpyDefault);   

    h_start.x = 3;
    h_start.y = 3;
    h_start.z = 3;
    h_end.x = h_AC_mx -3;
    h_end.y = h_AC_my -3;
    h_end.z = h_AC_mz -3;

    int3 size;
    size.x = h_end.x-h_start.x;
    size.y = h_end.y-h_start.y;
    size.z = h_end.z-h_start.z;

    cudaMemcpyToSymbol(AC_mx, &h_AC_mx, sizeof(int), 0, cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(AC_my, &h_AC_my, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(AC_mz, &h_AC_mz, sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(AC_inv_dsx, &h_AC_inv_dsx, sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(AC_inv_dsy, &h_AC_inv_dsy, sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(AC_inv_dsz, &h_AC_inv_dsz, sizeof(double), 0, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(start, &h_start, sizeof(int3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(end, &h_end, sizeof(int3), 0, cudaMemcpyHostToDevice);

    int lut[256];
    for (size_t i = 0; i < 256; i++)
    {
        lut[i] = i%xThreads;
    }
    cudaMemcpyToSymbol(xModLut, &lut, sizeof(int)*256, 0, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < 256; i++)
    {
        lut[i] = i%yThreads;
    }
    cudaMemcpyToSymbol(yModLut, &lut, sizeof(int)*256, 0, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < 256; i++)
    {
        lut[i] = i%zThreads;
    }
    cudaMemcpyToSymbol(zModLut, &lut, sizeof(int)*256, 0, cudaMemcpyHostToDevice);
    

    
    int xIndLut_h[(xThreads+6) * (yThreads+6) * (zThreads+6)];
    int yIndLut_h[(xThreads+6) * (yThreads+6) * (zThreads+6)];
    int zIndLut_h[(xThreads+6) * (yThreads+6) * (zThreads+6)];
    for (size_t i = 0; i < (xThreads+6) * (yThreads+6) * (zThreads+6); i += 1)
    {
        int xDiv = (i / (xThreads+6));
        xIndLut_h[i] = i % (xThreads+6);
        yIndLut_h[i] = (xDiv)%(yThreads+6);
        zIndLut_h[i] = (xDiv/(yThreads+6));
    }
    cudaMemcpyToSymbol(xIndLut, &xIndLut_h, sizeof(int)*(xThreads+6) * (yThreads+6) * (zThreads+6), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(yIndLut, &yIndLut_h, sizeof(int)*(xThreads+6) * (yThreads+6) * (zThreads+6), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(zIndLut, &zIndLut_h, sizeof(int)*(xThreads+6) * (yThreads+6) * (zThreads+6), 0, cudaMemcpyHostToDevice);

    // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);


    dim3 block = {xThreads,yThreads,zThreads};
    dim3 grid = {1+(size.x / block.x), 1+(size.y / block.y), 1+(size.z / block.z)};

    for (size_t i = 0; i < 10; i++)
    {
        auto start = std::chrono::steady_clock::now();
        kern<<<grid, block>>>(inBuf, outBuf);
        std::cout << cudaDeviceSynchronize() << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
            cudaMemcpy(hostBuf, outBuf, sizeof(AcReal)*count, cudaMemcpyDefault);

    std::cout << hostBuf[4241341] << " " << hostBuf[627389] << std::endl;

    double sum = 0;
    for (size_t i = 0; i < count; i++)
    {
        sum += hostBuf[i];
    }

    std::cout << std::setprecision(12) << sum << std::endl;
    
    }

    

 
    //cudaMemcpyToSymbol(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice);
}