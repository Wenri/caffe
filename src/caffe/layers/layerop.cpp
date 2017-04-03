#include <vector>
#include <boost/make_shared.hpp>
#include "caffe/filler.hpp"
#include "caffe/layers/layerop.hpp"
#include "caffe/util/math_functions.hpp"
#include "structured/interface/caffe/ProcessorTape_Caffe.h"
#include "structured/interface/caffe/TypedData_Caffe.h"

using namespace structured;

template<class SharedPointer> struct Holder {
    SharedPointer p;
    Holder(const SharedPointer &p) : p(p) {}
    Holder(const Holder &other): p(other.p) {}
    Holder(Holder &&other) : p(std::move(other.p)) {}
    void operator () (...) { p.reset(); }
};

namespace caffe {
    template <typename Dtype>
    LayerOpLayer<Dtype>::LayerOpLayer(const LayerParameter& param):
        Layer<Dtype>(param), env(param.layer_op_param().cmdline()),
        functor(functor::FunctorCaffe<Dtype>::acquireFunctor(env)){

        num_output = this->layer_param_.layer_op_param().num_output();
    }

    template <typename Dtype>
    LayerOpLayer<Dtype>::~LayerOpLayer() {
        this->blobs_.clear();
    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::initParams() {
        if(this->blobs_.size()) {
            shared_ptr<Filler<Dtype> > weight_filler(
                GetFiller<Dtype>(
                    this->layer_param_.layer_op_param().weight_filler()
                    ));
            for(auto&& var : this->blobs_)
                weight_filler->Fill(var.get());   // fill the weights
        }
        if (bias_term_) {
            vector<int> bias_shape(1, output_K);
            this->blobs_.emplace_back(
                boost::make_shared<Blob<Dtype>>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(
                GetFiller<Dtype>(
                    this->layer_param_.layer_op_param().bias_filler()
                    ));
            bias_filler->Fill(this->blobs_.back().get());
            this->param_propagate_down_.emplace_back(true);
        }
    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {

        bias_term_ = this->layer_param_.layer_op_param().bias_term();

        const int axis = bottom[0]->CanonicalAxisIndex(
            this->layer_param_.layer_op_param().axis());
        // Dimensions starting from "axis" are "flattened" into a single
        // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
        // and axis == 1, N inner products with dimension CHW are performed.
        input_K = bottom[0]->count(axis);
        LOG(INFO)<<"LayerOpLayerSetUp, N="<<num_output<<", K="<<input_K;

        if(this->blobs_.size() > 0) {
            LOG(INFO)<<"Vars already inited";
            processor.reset(functor->acquire(bottom, top));
        } else {
            if(bias_term_ || num_output) {
                // Check if we need to set up the weights
                auto learningVars = functor->allocateVars(num_output);
                this->blobs_.reserve(num_output + bias_term_);

                for (auto&& var : learningVars)
                    this->blobs_.emplace_back(
                        var.get(),
                        Holder<std::shared_ptr<Var_t<Dtype>>>(var)
                        );

                this->param_propagate_down_.resize(this->blobs_.size(), true);
            }
            processor.reset(functor->acquire(bottom, top));
            // parameter initialization
            this->initParams();
        }

    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {

        // Figure out the dimensions
        const int axis = bottom[0]->CanonicalAxisIndex(
            this->layer_param_.layer_op_param().axis());
        const int new_K = bottom[0]->count(axis);
        CHECK_EQ(input_K, new_K)
            << "Input size incompatible with inner product parameters.";

        // The first "axis" dimensions are independent inner products; the total
        // number of these is M_, the product over these dimensions.
        batch_size = bottom[0]->count(0, axis);
        // The top shape will be the bottom shape with the flattened axes dropped,
        // and replaced by a single axis with dimension num_output (N_).
        functor->load(bottom, top);
        output_K = top[0]->count(axis);

        if (bias_term_) {
            // Set up the bias multiplier
            vector<int> bias_shape(1, batch_size);
            bias_multiplier_.Reshape(bias_shape);
            caffe_set(batch_size, Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

        LOG(INFO)<<"Fwd Method called.\n";
        (*functor)(bottom, top);
        if (bias_term_) {
            Dtype* top_data = top[0]->mutable_cpu_data();
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                  batch_size, output_K, 1, (Dtype)1.,
                                  bias_multiplier_.cpu_data(),
                                  this->blobs_[1]->cpu_data(),
                                  (Dtype)1., top_data);
        }
    }

    template <typename Dtype>
    void LayerOpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {

        LOG(INFO)<<"Back Method called.\n";

        (*functor)(top, propagate_down, bottom);
        if (this->param_propagate_down_.size() &&
            this->param_propagate_down_[0]) {
            // Gradient with respect to weight
        }
        if (bias_term_ && this->param_propagate_down_[1]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bias
            caffe_cpu_gemv<Dtype>(CblasTrans, batch_size, output_K,
                                  (Dtype)1., top_diff,
                                  bias_multiplier_.cpu_data(), (Dtype)1.,
                                  this->blobs_[1]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
            // Gradient with respect to bottom data
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(LayerOpLayer);
#endif

    INSTANTIATE_CLASS(LayerOpLayer);
    REGISTER_LAYER_CLASS(LayerOp);

}  // namespace caffe
