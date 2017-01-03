#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/layerop.hpp"
#include "caffe/util/math_functions.hpp"
#include "structured/lib/ProcessorTape.h"
#include "structured/interface/caffe/TypedData_Caffe.h"
#include "structured/interface/caffe/layerop.hpp"
#include "structured/lib/AdjointPoint.h"

using namespace structured;

namespace caffe {

    template <typename Dtype>
    void LayerOpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        (*functor)(bottom, top);
        if (bias_term_) {
            Dtype* top_data = top[0]->mutable_gpu_data();
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                  batch_size, output_K, 1, (Dtype)1.,
                                  bias_multiplier_.gpu_data(),
                                  this->blobs_[1]->gpu_data(),
                                  (Dtype)1., top_data);
        }
    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
        (*functor)(top, propagate_down, bottom);
        if (this->param_propagate_down_[0]) {
            // Gradient with respect to weight
        }
        if (bias_term_ && this->param_propagate_down_[1]) {
            const Dtype* top_diff = top[0]->gpu_diff();
            // Gradient with respect to bias
            caffe_gpu_gemv<Dtype>(CblasTrans, batch_size, output_K,
                                  (Dtype)1., top_diff,
                                  bias_multiplier_.gpu_data(), (Dtype)1.,
                                  this->blobs_[1]->mutable_gpu_diff());
        }
        if (propagate_down[0]) {
            // Gradient with respect to bottom data
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(LayerOpLayer);

}  // namespace caffe
