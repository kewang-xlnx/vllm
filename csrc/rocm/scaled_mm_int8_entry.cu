// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct Int8GemmTensor
{
    Int8GemmTensor(float scale): scale_(scale){};

    template <typename E, typename C>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float>(
        ck::half_t& e, const float& c) const
    {
        const float x0_f = c * scale_;

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int>(
        ck::half_t& e, const int& c) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * scale_;

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int>(
        ck::bhalf_t& e, const int& c) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * scale_;

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<float, int>(
        float& e, const int& c) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * scale_;

        e = ck::type_convert<float>(x0_f);
    }
    float scale_;
};

at::Tensor int8_tensorwise_gemm(torch::Tensor XQ, torch::Tensor WQ, double scale)
{
    bool time_kernel     = false;
    ck::index_t M = XQ.size(0);
    ck::index_t N = WQ.size(0);
    ck::index_t K = XQ.size(1);

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    //ck::index_t KBatch = 1;

    TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
    TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

    auto Y = at::empty({M, N}, XQ.options().dtype(at::kHalf));
    
    using I8  = int8_t;
    using I32 = int;
    //using F16 = ck::half_t;
    
    using FT = ck::half_t;;
    // at::ScalarType dtype_y = Y.scalar_type();
    // if (dtype_y == at::ScalarType::Half) {
    //     using FT = ck::half_t;
    // } 
    // else if (dtype_y == at::ScalarType::BFloat16) {
    //     using FT = ck::bhalf_t;
    // } 
    // else if (dtype_y == at::ScalarType::Float) {
    //     using FT = float;
    // }
    // else {
    //     throw std::runtime_error("Unsupported ScalarType for type mapping!");
    // }
    
    using A0DataType       = I8;
    using B0DataType       = I8;
    using AccDataType      = I32;
    using CShuffleDataType = I32;
    using DsDataType       = ck::Tuple<>;
    using EDataType        = FT;

    using A0Layout = Row;
    using B0Layout = Col;
    using DsLayout = ck::Tuple<>;
    using ELayout  = Row;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = Int8GemmTensor;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNPadding;
    static constexpr auto LoopSched = ck::make_default_loop_scheduler();
    static constexpr auto PipelineVer = ck::PipelineVersion::v1;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
        // clang-format off
    ///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
    ///######|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
    ///######|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
    ///######|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
    ///###### RRR
        ///<      Row,      Row, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,   4,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;
    ///###### RCR
            <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,  16,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,     S<4, 64, 1>,    S<1, 0, 2>,     S<1, 0, 2>,             2,              16,             16,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;
    // clang-format on
    // using DeviceOpInstance =
    //   ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
    //       Row,
    //       Col,
    //       ck::Tuple<>, // D Layouts
    //       ELayout, // Output Layouts
    //       A0DataType,
    //       B0DataType,
    //       AccDataType,
    //       CShuffleDataType,
    //       ck::Tuple<>, // D Datatypes
    //       EDataType, // Output datatype
    //       AElementOp,
    //       BElementOp,
    //       CDEElementOp,
    //       GemmSpec, // Kernel Schedule.
    //       1, // Prefetch stage
    //       256, // Block size
    //       256, // M per block
    //       128, // N per block
    //       64, // K per block
    //       16, // AK1
    //       16, // BK1
    //       32, // M Per Xdl
    //       32, // N Per Xdl
    //       4, // Mxdl per wave
    //       2, // Nxdl per wave
    //       S<4, 64, 1>, // ABlockTransfer Threadcluster K0_M_K1
    //       S<1, 0, 2>, // ABlockTransfer ThreadCluster ArrangeOrder
    //       S<1, 0, 2>, // ABlockTransfer SrcAccessOrder
    //       2, // ABlockTransfer SrcVectorDim
    //       16, // ABlockTransfer SrcScalar PerVector
    //       16, // ABlockTransfer DstScalar PerVector_K1
    //       1, // ABlockLds AddExtraM
    //       S<4, 64, 1>, // BBlockTransfer ThreadCluster K0_N_K1
    //       S<1, 0, 2>, // BBlockTransfer ThreadCluster ArrangeOrder
    //       S<1, 0, 2>, // BBlockTransfer SrcAccess Order
    //       2, // BBlockTransfer SrcVectorDim
    //       16, // BBlockTransfer SrcScalarPerVector
    //       16, // BBlockTransfer DstScalar PerVector_K1
    //       1, // BBlockLds AddExtraN
    //       1, // CShuffle MXdlPerWave PerShuffle
    //       1, // CShuffle NXdlPerWave PerShuffle
    //       S<1, 32, 1, 8>, // CBlockTransferClusterLengths
    //       16, // CBlockTransfer ScalarPerVector
    //       LoopSched, // Loop Scheduler
    //       PipelineVer, // Pipeline version
    //       I8>; // Compute datatype

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{float(scale)};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();

    auto argument =
        device_op.MakeArgument(reinterpret_cast<A0DataType*>(XQ.data_ptr()),
                               reinterpret_cast<B0DataType*>(WQ.data_ptr()),
                               std::array<const void*, 0>{},
                               reinterpret_cast<EDataType*>(Y.data_ptr()),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, 0>{},
                               StrideE,
                               1,
                               a_element_op,
                               b_element_op,
                               cde_element_op);
    auto stream = at::cuda::getCurrentHIPStream().stream();
    invoker.Run(argument, StreamConfig{stream, false});
    //invoker.Run(argument, StreamConfig{nullptr, time_kernel, 20, 50});

    return Y;
}
