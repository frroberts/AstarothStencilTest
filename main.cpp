
#include <iostream>
#include <chrono>

#include <cuda_runtime_api.h>

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



typedef double AcReal;
typedef double3 AcReal3;

typedef struct {
    AcReal3 row[3];
} AcMatrix;

struct AcRealData
{
    double value;
    double3 gradient;
    AcMatrix hessian;
};


__constant__ int AC_mx;
__constant__ int AC_my;
__constant__ int AC_mz;
__constant__ double AC_inv_dsx;
__constant__ double AC_inv_dsy;
__constant__ double AC_inv_dsz;
__constant__ int3 start;
__constant__ int3 end;

static __device__
size_t acVertexBufferIdx(const int i, const int j, const int k)
{
    return i +                          //
           j * AC_mx + //
           k * AC_mx * AC_my;
}

static __device__ int
IDX(const int i, const int j, const int k)
{
    return acVertexBufferIdx(i, j, k);
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
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z)];
    }
    return first_derivative(pencil, AC_inv_dsx);
}

static __device__ __forceinline__
AcReal derxx(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z)];
    }
    return second_derivative(pencil, AC_inv_dsx);
}

static __device__ __forceinline__
AcReal derxy(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y + (6) / 2 - offset, vertexIdx.z)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal derxz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = arr[IDX(vertexIdx.x + offset - (6) / 2, vertexIdx.y, vertexIdx.z + (6) / 2 - offset)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsx, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal dery(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    return first_derivative(pencil, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal deryy(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z)];
    }
    return second_derivative(pencil, AC_inv_dsy);
}

static __device__ __forceinline__
AcReal deryz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil_a[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_a[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + offset - (6) / 2)];
    }
    AcReal pencil_b[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil_b[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y + offset - (6) / 2, vertexIdx.z + (6) / 2 - offset)];
    }
    return cross_derivative(pencil_a, pencil_b, AC_inv_dsy, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal derz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    return first_derivative(pencil, AC_inv_dsz);
}

static __device__ __forceinline__
AcReal derzz(int3 vertexIdx, const AcReal *__restrict__ arr) {
    AcReal pencil[(6) + 1];
    for (int offset = 0; offset < (6) + 1; ++offset) {
        pencil[IDX(offset)] = arr[IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z + offset - (6) / 2)];
    }
    return second_derivative(pencil, AC_inv_dsz);
}


// access functions

static __device__ __forceinline__
AcReal preprocessed_value(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex) {
    return vertex[IDX(vertexIdx)];
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
AcRealData read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf) {
    AcRealData data;


    data.value = preprocessed_value(vertexIdx, globalVertexIdx, buf);
    data.gradient = preprocessed_gradient(vertexIdx, globalVertexIdx, buf);
    data.hessian = preprocessed_hessian(vertexIdx, globalVertexIdx, buf);
    return data;
}


// kernel that does the call

__global__ void kern(AcReal* __restrict__ buf, AcReal* __restrict__ bufOut){
    //
    const int3 vertexIdx       = (int3){threadIdx.x + blockIdx.x * blockDim.x + start.x,
                                  threadIdx.y + blockIdx.y * blockDim.y + start.y,
                                  threadIdx.z + blockIdx.z * blockDim.z + start.z};

    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
        return;

    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

    AcRealData dat = read_data(vertexIdx, vertexIdx, buf);

// sum it all together to stop it being optimized
    bufOut[idx] = dat.value + dat.gradient.x + dat.gradient.y + dat.gradient.z + 
        dat.hessian.row[0].x + dat.hessian.row[0].y + dat.hessian.row[0].z + 
        dat.hessian.row[1].x + dat.hessian.row[1].y + dat.hessian.row[1].z + 
        dat.hessian.row[2].x + dat.hessian.row[2].y + dat.hessian.row[2].z;
}

__global__ void filler(AcReal* __restrict__ buf, AcReal* __restrict__ bufOut, int count)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    if(idx >= count)
        return;
    buf[idx] = 1;// + threadIdx.x/120.0;
    bufOut[idx] = 0;
}

int main() {
    //

    int h_AC_mx = 256;
    int h_AC_my = 256;
    int h_AC_mz = 256;
    double h_AC_inv_dsx = 1;
    double h_AC_inv_dsy = 1;
    double h_AC_inv_dsz = 1;
    int3 h_start;
    int3 h_end;

    int count = h_AC_mx * h_AC_my * h_AC_mz;

    AcReal *inBuf;
    AcReal *outBuf;
    AcReal *hostBuf = new AcReal[count];

    cudaMalloc((void**)&inBuf, count * sizeof(AcReal));
    cudaMalloc((void**)&outBuf, count * sizeof(AcReal));

    filler<<<1+(count/512), 512>>>(inBuf, outBuf, count);
    
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


    dim3 block = {32,4,4};
    dim3 grid = {1+(size.x / block.x), 1+(size.y / block.y), 1+(size.z / block.z)};

    for (size_t i = 0; i < 10; i++)
    {
        auto start = std::chrono::steady_clock::now();
        kern<<<grid, block>>>(inBuf, outBuf);
        std::cout << cudaDeviceSynchronize() << std::endl;
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }

    
    cudaMemcpy(hostBuf, outBuf, sizeof(AcReal)*count, cudaMemcpyDefault);

    std::cout << hostBuf[12413431] << " " << hostBuf[627389] << std::endl;

    double sum = 0;
    for (size_t i = 0; i < count; i++)
    {
        sum += hostBuf[i];
    }

    std::cout << sum << std::endl;
    
 
    //cudaMemcpyToSymbol(d_mesh_info, &value, sizeof(value), offset, cudaMemcpyHostToDevice);
}