#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        // ReluLLaMA stores down_proj as [n_embd, n_ff]. Convert the activation
        // to FP16 and let cuBLAS consume the stored orientation directly.
        GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
        GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);

        ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(), ggml_nelements(src1));
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1_d, src1_f16.get(), ggml_nelements(src1), stream);

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, src1_cublas_op,
                ne0, ne1, ne01,
                &alpha, src0->data,     CUDA_R_16F, ne0,
                        src1_f16.get(), CUDA_R_16F, ldb,
                &beta,  dst_d,          CUDA_R_32F, ne0,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        return;
    }

    const float * src0_d = (const float *) src0->data;
    const int64_t lda = nb01 / sizeof(float);

    // data strides in dimensions 2/3
    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }
}
