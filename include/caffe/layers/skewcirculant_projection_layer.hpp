#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SkewcirculantProjectionLayer : public Layer<Dtype> {
 public:
  explicit SkewcirculantProjectionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SkewcirculantProjection"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  // void FillData(const int n, const int a, const int b, const Dtype* x, Dtype* y);
  // void ComplexMulReal(const int n, const complex<Dtype>* a, const Dtype* b, complex<Dtype>* y);
  // void FFTSkewCirc(const int Trans, const int IsVBatch, const Dtype* v, const Dtype* x,  Dtype* result);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  bool flip_term_;
  Blob<Dtype> bias_multiplier_;
  Blob<complex<Dtype> > assist_;

private:
  Dtype getFlipInput(const Dtype* input, int index);
  void initFlipParams();
  void initBiasParams();
  void initCircParams();
  void reshapeBuffer();
  
  Blob<Dtype> data_buffer_;
  Blob<complex<Dtype> > conv_buffer_;
  
  Blob<Dtype> weight_buffer_;
  Blob<complex<Dtype> > param_buffer_;
 

};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
