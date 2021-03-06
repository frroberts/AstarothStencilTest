// should replace user_kernels.h
// kernel launch needs to be done with a threadblock size of {xThreadx, yThreads, zThreads} so the default autotuner needs to be bypassed

#pragma once
constexpr int xThreads = 16;
constexpr int yThreads = 8;
constexpr int zThreads = 4;

static const int AC_nx_DEFAULT_VALUE = 0;
static const int AC_ny_DEFAULT_VALUE = 0;
static const int AC_nz_DEFAULT_VALUE = 0;
static const int AC_mx_DEFAULT_VALUE = 0;
static const int AC_my_DEFAULT_VALUE = 0;
static const int AC_mz_DEFAULT_VALUE = 0;
static const int AC_nx_min_DEFAULT_VALUE = 0;
static const int AC_ny_min_DEFAULT_VALUE = 0;
static const int AC_nz_min_DEFAULT_VALUE = 0;
static const int AC_nx_max_DEFAULT_VALUE = 0;
static const int AC_ny_max_DEFAULT_VALUE = 0;
static const int AC_nz_max_DEFAULT_VALUE = 0;
static const int AC_mxy_DEFAULT_VALUE = 0;
static const int AC_nxy_DEFAULT_VALUE = 0;
static const int AC_nxyz_DEFAULT_VALUE = 0;
static const int AC_bc_type_bot_x_DEFAULT_VALUE = 0;
static const int AC_bc_type_bot_y_DEFAULT_VALUE = 0;
static const int AC_bc_type_bot_z_DEFAULT_VALUE = 0;
static const int AC_bc_type_top_x_DEFAULT_VALUE = 0;
static const int AC_bc_type_top_y_DEFAULT_VALUE = 0;
static const int AC_bc_type_top_z_DEFAULT_VALUE = 0;
static const int3 AC_global_grid_n_DEFAULT_VALUE = make_int3(0, 0, 0);
static const int3 AC_multigpu_offset_DEFAULT_VALUE = make_int3(0, 0, 0);
static const AcReal AC_dt_DEFAULT_VALUE = (AcReal)0.0;
typedef struct
{
    AcReal value;
    AcReal3 gradient;
    AcMatrix hessian;
} AcRealData;
typedef struct
{
    AcRealData x;
    AcRealData y;
    AcRealData z;
} AcReal3Data;
static __device__ AcRealData read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf[], const int handle, AcReal *__restrict__ sharedBuf, int3 end, int3 start);
static __device__ __forceinline__ AcReal3Data read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf[], const int3 &handle, AcReal *__restrict__ sharedBuf, int3 end, int3 start);
static __device__ AcReal value(const AcRealData &data);
static __device__ AcReal3 gradient(const AcRealData &data);
static __device__ AcMatrix hessian(const AcRealData &data);
static AcReal AC_dsx_DEFAULT_VALUE = AcReal(0.04908738521);
static AcReal AC_dsy_DEFAULT_VALUE = AcReal(0.04908738521);
static AcReal AC_dsz_DEFAULT_VALUE = AcReal(0.04908738521);
static AcReal AC_inv_dsx_DEFAULT_VALUE = AcReal(1.0) / AC_dsx_DEFAULT_VALUE;
static AcReal AC_inv_dsy_DEFAULT_VALUE = AcReal(1.0) / AC_dsy_DEFAULT_VALUE;
static AcReal AC_inv_dsz_DEFAULT_VALUE = AcReal(1.0) / AC_dsz_DEFAULT_VALUE;
static __device__ __forceinline__ AcReal
first_derivative(AcReal pencil[], AcReal inv_ds)
{
    AcReal coefficients[] = {
        0, AcReal(3.0) / AcReal(4.0), -AcReal(3.0) / AcReal(20.0), AcReal(1.0) / AcReal(60.0)};
    AcReal res = 0;
    for (int i = 1;
         i <= ((6) / 2);
         ++i)
    {
        res += coefficients[IDX(i)] * (pencil[IDX(((6) / 2) + i)] - pencil[IDX(((6) / 2) - i)]);
    }
    return res * inv_ds;
}
static __device__ __forceinline__ AcReal
second_derivative(AcReal pencil[], AcReal inv_ds)
{
    AcReal coefficients[] = {
        -AcReal(49.0) / AcReal(18.0), AcReal(3.0) / AcReal(2.0), -AcReal(3.0) / AcReal(20.0), AcReal(1.0) / AcReal(90.0)};
    AcReal res = coefficients[IDX(0)] * pencil[IDX(((6) / 2))];
    for (int i = 1;
         i <= ((6) / 2);
         ++i)
    {
        res += coefficients[IDX(i)] * (pencil[IDX(((6) / 2) + i)] + pencil[IDX(((6) / 2) - i)]);
    }
    return res * inv_ds * inv_ds;
}
static __device__ __forceinline__ AcReal
cross_derivative(AcReal pencil_a[], AcReal pencil_b[], AcReal inv_ds_a, AcReal inv_ds_b)
{
    AcReal fac = AcReal(1.0) / AcReal(720.0);
    AcReal coefficients[] = {
        AcReal(0.0) * fac, AcReal(270.0) * fac, -AcReal(27.0) * fac, AcReal(2.0) * fac};
    AcReal res = AcReal(0.0);
    for (int i = 1;
         i <= ((6) / 2);
         ++i)
    {
        res += coefficients[IDX(i)] * (pencil_a[IDX(((6) / 2) + i)] + pencil_a[IDX(((6) / 2) - i)] - pencil_b[IDX(((6) / 2) + i)] - pencil_b[IDX(((6) / 2) - i)]);
    }
    return res * inv_ds_a * inv_ds_b;
}




static __device__
size_t acVertexBufferIdx_shared(const int i, const int j, const int k)
{
    return i +                          //
           j * (xThreads+6) + //
           k * (xThreads+6) * (yThreads+6);
}

static __device__ int
IDX_shared(const int i, const int j, const int k)
{
    return acVertexBufferIdx_shared(i, j, k);
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

    AcReal ret = arr[IDX_shared(vertexIdxReal.x + vertexOffsets.x, vertexIdxReal.y + vertexOffsets.y, vertexIdxReal.z + vertexOffsets.z)];
    
    return ret;

#else
    // if not shared
    return arr[IDX(vertexIdxReal.x + vertexOffsets.x, vertexIdxReal.y + vertexOffsets.y, vertexIdxReal.z + vertexOffsets.z)];
#endif
}



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

static __device__ __forceinline__
AcReal preprocessed_value(const int3 &vertexIdx, const int3 &globalVertexIdx, const AcReal *__restrict__ vertex) {
    return getData(vertexIdx, {0,0,0}, vertex);//vertex[IDX(vertexIdx)];
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

    data.hessian.row[IDX(1)].x = data.hessian.row[IDX(0)].y;
    data.hessian.row[IDX(2)].x = data.hessian.row[IDX(0)].z;
    data.hessian.row[IDX(2)].y = data.hessian.row[IDX(1)].z;
    return;
}

static __device__ __forceinline__ AcReal3
value(const AcReal3Data &uu)
{
    return (AcReal3){
        value(uu.x), value(uu.y), value(uu.z)};
}
static __device__ __forceinline__ AcReal3
preprocessed_gradient(GEN_PREPROCESSED_PARAM_BOILERPLATE, const AcReal *__restrict__ vertex)
{
    assert(DCONST(AC_dsx) > 0);
    assert(DCONST(AC_dsy) > 0);
    assert(DCONST(AC_dsz) > 0);
    assert(DCONST(AC_inv_dsx) > 0);
    assert(DCONST(AC_inv_dsy) > 0);
    assert(DCONST(AC_inv_dsz) > 0);
    return (AcReal3){
        derx(vertexIdx, vertex), dery(vertexIdx, vertex), derz(vertexIdx, vertex)};
}
static __device__ __forceinline__ AcMatrix
preprocessed_hessian(GEN_PREPROCESSED_PARAM_BOILERPLATE, const AcReal *__restrict__ vertex)
{
    assert(DCONST(AC_dsx) > 0);
    assert(DCONST(AC_dsy) > 0);
    assert(DCONST(AC_dsz) > 0);
    assert(DCONST(AC_inv_dsx) > 0);
    assert(DCONST(AC_inv_dsy) > 0);
    assert(DCONST(AC_inv_dsz) > 0);
    AcMatrix mat;
    mat.row[IDX(0)] = (AcReal3){
        derxx(vertexIdx, vertex), derxy(vertexIdx, vertex), derxz(vertexIdx, vertex)};
    mat.row[IDX(1)] = (AcReal3){
        mat.row[IDX(0)].y, deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    mat.row[IDX(2)] = (AcReal3){
        mat.row[IDX(0)].z, mat.row[IDX(1)].z, derzz(vertexIdx, vertex)};
    return mat;
}
static __device__ __forceinline__ AcReal
laplace(const AcRealData &data)
{
    return hessian(data).row[IDX(0)].x + hessian(data).row[IDX(1)].y + hessian(data).row[IDX(2)].z;
}
static __device__ __forceinline__ AcReal
divergence(const AcReal3Data &vec)
{
    return gradient(vec.x).x + gradient(vec.y).y + gradient(vec.z).z;
}
static __device__ __forceinline__ AcReal3
laplace_vec(const AcReal3Data &vec)
{
    return (AcReal3){
        laplace(vec.x), laplace(vec.y), laplace(vec.z)};
}
static __device__ __forceinline__ AcReal3
curl(const AcReal3Data &vec)
{
    return (AcReal3){
        gradient(vec.z).y - gradient(vec.y).z, gradient(vec.x).z - gradient(vec.z).x, gradient(vec.y).x - gradient(vec.x).y};
}
static __device__ __forceinline__ AcReal3
gradient_of_divergence(const AcReal3Data &vec)
{
    return (AcReal3){
        hessian(vec.x).row[IDX(0)].x + hessian(vec.y).row[IDX(0)].y + hessian(vec.z).row[IDX(0)].z, hessian(vec.x).row[IDX(1)].x + hessian(vec.y).row[IDX(1)].y + hessian(vec.z).row[IDX(1)].z, hessian(vec.x).row[IDX(2)].x + hessian(vec.y).row[IDX(2)].y + hessian(vec.z).row[IDX(2)].z};
}
static __device__ __forceinline__ AcMatrix
stress_tensor(const AcReal3Data &vec)
{
    AcMatrix S;
    S.row[IDX(0)].x = (AcReal(2.0) / AcReal(3.0)) * gradient(vec.x).x - (AcReal(1.0) / AcReal(3.0)) * (gradient(vec.y).y + gradient(vec.z).z);
    S.row[IDX(0)].y = (AcReal(1.0) / AcReal(2.0)) * (gradient(vec.x).y + gradient(vec.y).x);
    S.row[IDX(0)].z = (AcReal(1.0) / AcReal(2.0)) * (gradient(vec.x).z + gradient(vec.z).x);
    S.row[IDX(1)].y = (AcReal(2.0) / AcReal(3.0)) * gradient(vec.y).y - (AcReal(1.0) / AcReal(3.0)) * (gradient(vec.x).x + gradient(vec.z).z);
    S.row[IDX(1)].z = (AcReal(1.0) / AcReal(2.0)) * (gradient(vec.y).z + gradient(vec.z).y);
    S.row[IDX(2)].z = (AcReal(2.0) / AcReal(3.0)) * gradient(vec.z).z - (AcReal(1.0) / AcReal(3.0)) * (gradient(vec.x).x + gradient(vec.y).y);
    S.row[IDX(1)].x = S.row[IDX(0)].y;
    S.row[IDX(2)].x = S.row[IDX(0)].z;
    S.row[IDX(2)].y = S.row[IDX(1)].z;
    return S;
}
static __device__ __forceinline__ AcReal
contract(const AcMatrix mat)
{
    AcReal res = 0;
    for (int i = 0;
         i < 3;
         ++i)
    {
        res += dot(mat.row[IDX(i)], mat.row[IDX(i)]);
    }
    return res;
}
static __device__ __forceinline__ AcReal
length(const AcReal3 vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}
static __device__ __forceinline__ AcReal
reciprocal_len(const AcReal3 vec)
{
    return rsqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}
static __device__ __forceinline__ AcReal3
normalized(const AcReal3 vec)
{
    const AcReal inv_len = reciprocal_len(vec);
    return inv_len * vec;
}
static int AC_max_steps_DEFAULT_VALUE;
static int AC_save_steps_DEFAULT_VALUE;
static int AC_bin_steps_DEFAULT_VALUE;
static int AC_start_step_DEFAULT_VALUE;
static AcReal AC_max_time_DEFAULT_VALUE;
static AcReal AC_dsmin_DEFAULT_VALUE;
static AcReal AC_xlen_DEFAULT_VALUE;
static AcReal AC_ylen_DEFAULT_VALUE;
static AcReal AC_zlen_DEFAULT_VALUE;
static AcReal AC_xorig_DEFAULT_VALUE;
static AcReal AC_yorig_DEFAULT_VALUE;
static AcReal AC_zorig_DEFAULT_VALUE;
static AcReal AC_unit_density_DEFAULT_VALUE;
static AcReal AC_unit_velocity_DEFAULT_VALUE;
static AcReal AC_unit_length_DEFAULT_VALUE;
static AcReal AC_unit_magnetic_DEFAULT_VALUE;
static AcReal AC_star_pos_x_DEFAULT_VALUE;
static AcReal AC_star_pos_y_DEFAULT_VALUE;
static AcReal AC_star_pos_z_DEFAULT_VALUE;
static AcReal AC_M_star_DEFAULT_VALUE;
static AcReal AC_sink_pos_x_DEFAULT_VALUE;
static AcReal AC_sink_pos_y_DEFAULT_VALUE;
static AcReal AC_sink_pos_z_DEFAULT_VALUE;
static AcReal AC_M_sink_DEFAULT_VALUE;
static AcReal AC_M_sink_init_DEFAULT_VALUE;
static AcReal AC_M_sink_Msun_DEFAULT_VALUE;
static AcReal AC_soft_DEFAULT_VALUE;
static AcReal AC_accretion_range_DEFAULT_VALUE;
static AcReal AC_switch_accretion_DEFAULT_VALUE;
static AcReal AC_cdt_DEFAULT_VALUE;
static AcReal AC_cdtv_DEFAULT_VALUE;
static AcReal AC_cdts_DEFAULT_VALUE;
static AcReal AC_nu_visc_DEFAULT_VALUE;
static AcReal AC_cs_sound_DEFAULT_VALUE = AcReal(1.0);
static AcReal AC_eta_DEFAULT_VALUE;
static AcReal AC_mu0_DEFAULT_VALUE;
static AcReal AC_cp_sound_DEFAULT_VALUE;
static AcReal AC_gamma_DEFAULT_VALUE;
static AcReal AC_cv_sound_DEFAULT_VALUE;
static AcReal AC_lnT0_DEFAULT_VALUE;
static AcReal AC_lnrho0_DEFAULT_VALUE;
static AcReal AC_zeta_DEFAULT_VALUE;
static AcReal AC_trans_DEFAULT_VALUE;
static AcReal AC_nu_shock_DEFAULT_VALUE;
static AcReal AC_bin_save_t_DEFAULT_VALUE;
static AcReal AC_ampl_lnrho_DEFAULT_VALUE;
static AcReal AC_ampl_uu_DEFAULT_VALUE;
static AcReal AC_angl_uu_DEFAULT_VALUE;
static AcReal AC_lnrho_edge_DEFAULT_VALUE;
static AcReal AC_lnrho_out_DEFAULT_VALUE;
static AcReal AC_ampl_aa_DEFAULT_VALUE;
static AcReal AC_init_k_wave_DEFAULT_VALUE;
static AcReal AC_init_sigma_hel_DEFAULT_VALUE;
static AcReal AC_forcing_magnitude_DEFAULT_VALUE;
static AcReal AC_relhel_DEFAULT_VALUE;
static AcReal AC_kmin_DEFAULT_VALUE;
static AcReal AC_kmax_DEFAULT_VALUE;
static AcReal AC_forcing_phase_DEFAULT_VALUE;
static AcReal AC_k_forcex_DEFAULT_VALUE;
static AcReal AC_k_forcey_DEFAULT_VALUE;
static AcReal AC_k_forcez_DEFAULT_VALUE;
static AcReal AC_kaver_DEFAULT_VALUE;
static AcReal AC_ff_hel_rex_DEFAULT_VALUE;
static AcReal AC_ff_hel_rey_DEFAULT_VALUE;
static AcReal AC_ff_hel_rez_DEFAULT_VALUE;
static AcReal AC_ff_hel_imx_DEFAULT_VALUE;
static AcReal AC_ff_hel_imy_DEFAULT_VALUE;
static AcReal AC_ff_hel_imz_DEFAULT_VALUE;
static AcReal AC_G_const_DEFAULT_VALUE;
static AcReal AC_GM_star_DEFAULT_VALUE;
static AcReal AC_unit_mass_DEFAULT_VALUE;
static AcReal AC_sq2GM_star_DEFAULT_VALUE;
static AcReal AC_cs2_sound_DEFAULT_VALUE = AC_cs_sound_DEFAULT_VALUE * AC_cs_sound_DEFAULT_VALUE;
;
;
;
;
;
;
;
;
static __device__ __forceinline__ AcMatrix
gradients(const AcReal3Data &uu)
{
    return (AcMatrix){
        gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}
static __device__ __forceinline__ AcReal
continuity(int3 globalVertexIdx, const AcReal3Data &uu, const AcRealData &lnrho, AcReal dt)
{
    return -dot(value(uu), gradient(lnrho)) - divergence(uu);
}
static __device__ __forceinline__ AcReal3
momentum(int3 globalVertexIdx, const AcReal3Data &uu, const AcRealData &lnrho, const AcRealData &ss, const AcReal3Data &aa, AcReal dt)
{
    const AcMatrix S = stress_tensor(uu);
    const AcReal cs2 = DCONST(AC_cs2_sound) * exp(DCONST(AC_gamma) * value(ss) / DCONST(AC_cp_sound) + (DCONST(AC_gamma) - 1) * (value(lnrho) - DCONST(AC_lnrho0)));
    const AcReal3 j = (AcReal(1.0) / DCONST(AC_mu0)) * (gradient_of_divergence(aa) - laplace_vec(aa));
    const AcReal3 B = curl(aa);
    const AcReal inv_rho = AcReal(1.0) / exp(value(lnrho));
    const AcReal3 mom = -mul(gradients(uu), value(uu)) - cs2 * ((AcReal(1.0) / DCONST(AC_cp_sound)) * gradient(ss) + gradient(lnrho)) + inv_rho * cross(j, B) + DCONST(AC_nu_visc) * (laplace_vec(uu) + (AcReal(1.0) / AcReal(3.0)) * gradient_of_divergence(uu) + AcReal(2.0) * mul(S, gradient(lnrho))) + DCONST(AC_zeta) * gradient_of_divergence(uu);
    return mom;
}
static __device__ __forceinline__ AcReal3
induction(const AcReal3Data &uu, const AcReal3Data &aa)
{
    const AcReal3 B = curl(aa);
    const AcReal3 lap = laplace_vec(aa);
    const AcReal3 ind = cross(value(uu), B) + DCONST(AC_eta) * lap;
    return ind;
}
static __device__ __forceinline__ AcReal
lnT(const AcRealData &ss, const AcRealData &lnrho)
{
    return DCONST(AC_lnT0) + DCONST(AC_gamma) * value(ss) / DCONST(AC_cp_sound) + (DCONST(AC_gamma) - AcReal(1.0)) * (value(lnrho) - DCONST(AC_lnrho0));
}
static __device__ __forceinline__ AcReal
heat_conduction(const AcRealData &ss, const AcRealData &lnrho)
{
    const AcReal inv_AC_cp_sound = AcReal(1.0) / DCONST(AC_cp_sound);
    const AcReal3 grad_ln_chi = -gradient(lnrho);
    const AcReal first_term = DCONST(AC_gamma) * inv_AC_cp_sound * laplace(ss) + (DCONST(AC_gamma) - AcReal(1.0)) * laplace(lnrho);
    const AcReal3 second_term = DCONST(AC_gamma) * inv_AC_cp_sound * gradient(ss) + (DCONST(AC_gamma) - AcReal(1.0)) * gradient(lnrho);
    const AcReal3 third_term = DCONST(AC_gamma) * (inv_AC_cp_sound * gradient(ss) + gradient(lnrho)) + grad_ln_chi;
    const AcReal chi = (AcReal(0.001)) / (exp(value(lnrho)) * DCONST(AC_cp_sound));
    return DCONST(AC_cp_sound) * chi * (first_term + dot(second_term, third_term));
}
static __device__ __forceinline__ AcReal
heating(const int i, const int j, const int k)
{
    return 1;
}
static __device__ __forceinline__ AcReal
entropy(const AcRealData &ss, const AcReal3Data &uu, const AcRealData &lnrho, const AcReal3Data &aa)
{
    const AcMatrix S = stress_tensor(uu);
    const AcReal inv_pT = AcReal(1.0) / (exp(value(lnrho)) * exp(lnT(ss, lnrho)));
    const AcReal3 j = (AcReal(1.0) / DCONST(AC_mu0)) * (gradient_of_divergence(aa) - laplace_vec(aa));
    const AcReal RHS = (0) - (0) + DCONST(AC_eta) * (DCONST(AC_mu0)) * dot(j, j) + AcReal(2.0) * exp(value(lnrho)) * DCONST(AC_nu_visc) * contract(S) + DCONST(AC_zeta) * exp(value(lnrho)) * divergence(uu) * divergence(uu);
    return -dot(value(uu), gradient(ss)) + inv_pT * RHS + heat_conduction(ss, lnrho);
}
static __device__ const int handle_lnrho(DCONST(VTXBUF_LNRHO));
static __device__ const int handle_out_lnrho(DCONST(VTXBUF_LNRHO));
static __device__ const int3 handle_uu = make_int3(DCONST(VTXBUF_UUX), DCONST(VTXBUF_UUY), DCONST(VTXBUF_UUZ));
static __device__ const int3 handle_out_uu = make_int3(DCONST(VTXBUF_UUX), DCONST(VTXBUF_UUY), DCONST(VTXBUF_UUZ));
static __device__ const int3 handle_aa = make_int3(DCONST(VTXBUF_AX), DCONST(VTXBUF_AY), DCONST(VTXBUF_AZ));
static __device__ const int3 handle_out_aa = make_int3(DCONST(VTXBUF_AX), DCONST(VTXBUF_AY), DCONST(VTXBUF_AZ));
static __device__ const int handle_ss(DCONST(VTXBUF_ENTROPY));
static __device__ const int handle_out_ss(DCONST(VTXBUF_ENTROPY));
template <int step_number>
static __global__ void
    solve(GEN_KERNEL_PARAM_BOILERPLATE)
{
    //GEN_KERNEL_BUILTIN_VARIABLES_BOILERPLATE();

    const int3 vertexIdx       = (int3){threadIdx.x + blockIdx.x * blockDim.x + start.x,           
                                  threadIdx.y + blockIdx.y * blockDim.y + start.y,           
                                  threadIdx.z + blockIdx.z * blockDim.z + start.z};          

    #ifdef SHAREDCACHE
    __shared__ AcReal sharedBuf[(xThreads+6)*(yThreads+6)*(zThreads+6)];
    #else
    AcReal *sharedBuf; // leave uninitialized 
    #endif

    const int3 globalVertexIdx = (int3){d_multigpu_offset.x + vertexIdx.x,                         
                                        d_multigpu_offset.y + vertexIdx.y,                         
                                        d_multigpu_offset.z + vertexIdx.z};                        
    (void)globalVertexIdx;                                                                         
    // we cant exit the threads since i need them to fill the shared buffer
    //if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
    //    return;
                                                                                  
    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&                   
           vertexIdx.z < DCONST(AC_nz_max));                                                       
                                                                                                   
    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&                 
           vertexIdx.z >= DCONST(AC_nz_min));                                                      
                                                                                                   
    const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);

    const AcRealData lnrho = read_data(vertexIdx, globalVertexIdx, buffer.in, handle_lnrho, sharedBuf, end, start);
    const AcReal3Data uu = read_data(vertexIdx, globalVertexIdx, buffer.in, handle_uu, sharedBuf, end, start);
    const AcReal3Data aa = read_data(vertexIdx, globalVertexIdx, buffer.in, handle_aa, sharedBuf, end, start);
    const AcRealData ss = read_data(vertexIdx, globalVertexIdx, buffer.in, handle_ss, sharedBuf, end, start);

    if (!(vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z))
    {
        AcReal out_lnrho = READ_OUT(handle_out_lnrho);
        AcReal3 out_uu = READ_OUT(handle_out_uu);
        AcReal3 out_aa = READ_OUT(handle_out_aa);
        AcReal out_ss = READ_OUT(handle_out_ss);
        AcReal dt = DCONST(AC_dt);
        out_lnrho = rk3(out_lnrho, lnrho, continuity(globalVertexIdx, uu, lnrho, dt), dt);
        out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
        out_uu = rk3(out_uu, uu, momentum(globalVertexIdx, uu, lnrho, ss, aa, dt), dt);
        out_ss = rk3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt);
        WRITE_OUT(handle_out_lnrho, out_lnrho);
        WRITE_OUT(handle_out_uu, out_uu);
        WRITE_OUT(handle_out_aa, out_aa);
        WRITE_OUT(handle_out_ss, out_ss);
    }
}

static __device__ __forceinline__
AcRealData read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf[], const int handle, AcReal *__restrict__ sharedBuf, int3 end, int3 start) {
    AcRealData data;

#ifdef SHAREDCACHE

    int idxLocal = threadIdx.x + (threadIdx.y * xThreads) + (threadIdx.z * xThreads * yThreads);
#ifdef FLATFILL

    for (size_t i = idxLocal; i < (xThreads+6) * (yThreads+6) * (zThreads+6); i += xThreads * yThreads * zThreads)
    {
        int xDiv = (i / (xThreads+6));
        int x = i % (xThreads+6);
        int y = (xDiv)%(yThreads+6);
        int z = (xDiv/(yThreads+6));
        //int sharedInd = x + (y * (xThreads+6)) + (z *(xThreads+6)*(yThreads+6));
        int targetX = (blockIdx.x * blockDim.x) + start.x + x -3;
        int targetY = (blockIdx.y * blockDim.y) + start.y + y -3;
        int targetZ = (blockIdx.z * blockDim.z) + start.z + z -3;
        
        if(targetX >= end.x+3 || targetY >= end.y+3 || targetZ >= end.z+3)
            continue;
        sharedBuf[i] = buf[handle][IDX(targetX, targetY, targetZ)];
    }

#else
    // yes these are ordered wrong ... avoids bank conflicts ?
    for (size_t x = threadIdx.x; x < xThreads+6; x += xThreads)
    {
        for (size_t y = threadIdx.y; y < yThreads+6; y += yThreads)
        {
            for (size_t z = threadIdx.z; z < zThreads+6; z += zThreads)
            {
                int sharedInd = x + (y * (xThreads+6)) + (z *(xThreads+6)*(yThreads+6));
                int targetX = (blockIdx.x * blockDim.x) + start.x + x -3;
                int targetY = (blockIdx.y * blockDim.y) + start.y + y -3;
                int targetZ = (blockIdx.z * blockDim.z) + start.z + z -3;
                if(targetX >= end.x+3 || targetY >= end.y+3 || targetZ >= end.z+3)
                    continue;
                sharedBuf[sharedInd] = buf[handle][IDX(targetX, targetY, targetZ)];
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
/*
static __device__ __forceinline__ AcRealData read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf[], const int handle)
{
    AcRealData data;
    data.value = preprocessed_value(vertexIdx, globalVertexIdx, buf[handle]);
    data.gradient = preprocessed_gradient(vertexIdx, globalVertexIdx, buf[handle]);
    data.hessian = preprocessed_hessian(vertexIdx, globalVertexIdx, buf[handle]);
    return data;
}
*/
static __device__ __forceinline__ AcReal value(const AcRealData &data)
{
    return data.value;
}
static __device__ __forceinline__ AcReal3 gradient(const AcRealData &data)
{
    return data.gradient;
}
static __device__ __forceinline__ AcMatrix hessian(const AcRealData &data)
{
    return data.hessian;
}
static __device__ __forceinline__ AcReal3Data read_data(const int3 &vertexIdx, const int3 &globalVertexIdx, AcReal *__restrict__ buf[], const int3 &handle, AcReal *__restrict__ sharedBuf, int3 end, int3 start)
{
    AcReal3Data data;
    data.x = read_data(vertexIdx, globalVertexIdx, buf, handle.x, sharedBuf, end, start);
    data.y = read_data(vertexIdx, globalVertexIdx, buf, handle.y, sharedBuf, end, start);
    data.z = read_data(vertexIdx, globalVertexIdx, buf, handle.z, sharedBuf, end, start);
    return data;
}
GEN_KERNEL_FUNC_HOOK(solve)
