#ifndef CAFFE_LAYER_OP_LAYER_HPP_
#define CAFFE_LAYER_OP_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "structured/interface/caffe/layerop.hpp"
#include "structured/lib/ProcessorBase.h"

namespace caffe {

/**
 * @brief Also known as a "layer-op" layer, modified from inner product
 *        layer with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
    template <typename Dtype>
    class LayerOpLayer : public Layer<Dtype> {
    public:
        explicit LayerOpLayer(const LayerParameter& param);
        ~LayerOpLayer();

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "LayerOpLayer"; }
        virtual inline int ExactNumBottomBlobs() const { return env.num_inputs; }
        virtual inline int ExactNumTopBlobs() const { return env.num_outputs; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom);

        void initParams();

        int num_output;
        int batch_size;
        int input_K;
        int output_K;

        bool bias_term_;
        Blob<Dtype> bias_multiplier_;
        structured::Environment env;
        std::unique_ptr<structured::ProcessorBase> processor;
        std::unique_ptr<structured::functor::FunctorCaffe<Dtype>> functor;
    };

}  // namespace caffe

#endif  // CAFFE_LAYER_OP_LAYER_HPP_
