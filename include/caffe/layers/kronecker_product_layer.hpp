#ifndef CAFFE_KRONECKER_PRODUCT_LAYER_HPP_
#define CAFFE_KRONECKER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an kronecker product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class KroneckerProductLayer : public Layer<Dtype> {
 public:
  explicit KroneckerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KroneckerProduct"; }
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

  vector<std::pair<int, int> > dims_;
  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> kpbuf_;

 private:
  void akpbx(const Dtype* A, const Dtype* B, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2);
  void dakpbx(const Dtype* top, const Dtype* B, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2);
  void akpdbx(const Dtype* top, const Dtype* A, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2);
  void akpbdx(const Dtype* top, const Dtype* A, const Dtype* B, Dtype *t, const int k1, const int d1, const int k2, const int d2);
  bool unitTest();
};

}  // namespace caffe

#endif  // CAFFE_KRONECKER_PRODUCT_LAYER_HPP_
