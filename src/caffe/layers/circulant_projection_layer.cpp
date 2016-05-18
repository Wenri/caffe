#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/circulant_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define pi 3.14159265358979323846

namespace caffe {

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::initCircParams() {
    // Intialize the weight
    vector<int> weight_shape(1, K_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.circulant_projection_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    this->param_propagate_down_[0] = true;
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::initBiasParams() {
    vector<int> bias_shape(1, N_);
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.circulant_projection_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());

    this->param_propagate_down_[1] = true;
}


template <typename Dtype>
void CirculantProjectionLayer<Dtype>::initFlipParams() {

    vector<int> weight_shape(1, K_);
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape));

    Dtype* flip_data = this->blobs_[2]->mutable_cpu_data();

    if (flip_term_) {
      auto func = [=](int i, bool r)
        {
          flip_data[i]=r?1.:1.; //:-1 change for test
        };

      caffe_rng_bernoulli<Dtype>(K_, (Dtype)0.5, func);
    }else{
      caffe_set(K_, Dtype(1), flip_data);
    }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::reshapeBuffer() {
    LOG(INFO)<<"Reshape"<<K_<<N_;
    vector<int> weight_shape(1, K_);
    this->weight_buffer_.Reshape(weight_shape);

    vector<int> data_shape(2);
    data_shape[0] = K_;
    data_shape[1] = K_;
    this->data_buffer_.Reshape(data_shape);

    vector<int> conv_shape(2);
    conv_shape[0] = M_;
    conv_shape[1] = K_;// changed from K_/2+1
    this->conv_buffer_.Reshape(conv_shape);


    vector<int> m_shape(2);
    m_shape[0] = M_;
    m_shape[1] = K_;
    this->m_buffer_.Reshape(m_shape);
}



template <typename Dtype>
void CirculantProjectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.circulant_projection_param().num_output();
  bias_term_ = this->layer_param_.circulant_projection_param().bias_term();
  flip_term_ = true;
  LOG(INFO)<<"CirculantProjectionLayerSetUp, N="<<num_output;
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
                                                       this->layer_param_.circulant_projection_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
    CHECK_LE(N_, K_) << "Currently only N<=K supported.";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    this->param_propagate_down_.resize(this->blobs_.size(), false);

    LOG(INFO)<<"Init Circ Projection";
    this->initCircParams();
    if (bias_term_) this->initBiasParams();
    this->initFlipParams();

    LOG(INFO)<<"Init Circ Done";
  }  // parameter initialization

  // Set up assist vector
  vector<int> assist_shape(1, K_);
  assist_.Reshape(assist_shape);
  complex<Dtype>* assist = assist_.mutable_cpu_data();

  LOG(INFO)<<"reshape_assist";
  assist[0] = 1.;
  assist[1] = 1.;//test for circ std::exp(std::complex<Dtype>(0, pi/K_)); //i*pi/K_
  for(int i =2; i<K_; i++)
    assist[i] = assist[1] * assist[i-1]; 

}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
			 this->layer_param_.circulant_projection_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
    top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  vector<int> bias_shape(1, M_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());

  this->reshapeBuffer();
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* R_vector = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Dtype* data_buffer = this->data_buffer_.mutable_cpu_data();

  // R * D * x
  for (int i =0; i<M_; i++)
    for (int j=0; j<K_; j++)
      (data_buffer+i*K_)[j] = this->blobs_[2]->cpu_data()[j]>(Dtype)0.5?(bottom_data+i*K_)[j] : -(bottom_data+i*K_)[j];


  this->FFTSkewCirc(0, 0, R_vector, data_buffer, data_buffer);
  FillData(M_,K_,N_,data_buffer,top_data);
 

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
     }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::ComplexMulReal(const int n, const complex<Dtype>* a, const Dtype* b, complex<Dtype>* y){
  for (int i=0; i<n; i++){
    y[i] = a[i] * b[i];
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::FFTSkewCirc(const int Trans, const int IsVBatch, const Dtype* v, const Dtype* x,  Dtype* result){

  const complex<Dtype>* assist = this->assist_.cpu_data();
  complex<Dtype>* m_buffer = this->m_buffer_.mutable_cpu_data();
  complex<Dtype>* v_buffer = (this->conv_buffer_.mutable_cpu_data());
  complex<Dtype>* x_buffer = (this->conv_buffer_.mutable_cpu_diff());
  Dtype* R_buffer = this->data_buffer_.mutable_cpu_diff();
  Dtype* R_vector = this->weight_buffer_.mutable_cpu_diff();
  
  // FFT version
  
  for (int i=0; i<M_; i++) ComplexMulReal(K_, assist, x+i*K_, m_buffer+i*K_);
  caffe_cpu_fft<complex<Dtype>>(M_, K_, m_buffer, x_buffer);

  for (int i=0; i<K_; i++) R_buffer[i] = v[i];
  if (!IsVBatch)
    for (int i=K_; i<M_*K_; i++) R_buffer[i] = v[i%K_];
 
  for (int i=0; i<M_;i++) ComplexMulReal(K_, assist, R_buffer+i*K_, m_buffer+i*K_);
  if (Trans) caffe_conj<Dtype>(M_*K_, m_buffer, m_buffer);
  caffe_cpu_fft<complex<Dtype>>(M_, K_, m_buffer, v_buffer);
  
    /*
    ComplexMulReal(K_, assist, v, m_buffer);
    if (Trans) caffe_conj<Dtype>(K_, m_buffer,m_buffer);
    caffe_cpu_fft<complex<Dtype>>(1, K_, m_buffer, v_buffer);
    for (int i=K_; i<M_*K_;i++) v_buffer[i] = v_buffer[i%K_];
    }*/

  caffe_mul<complex<Dtype>>(M_*K_, v_buffer, x_buffer, m_buffer);
  caffe_cpu_ifft<complex<Dtype>>(M_, K_, m_buffer, x_buffer);
  caffe_conj(K_, assist, v_buffer);
  for(int i=0; i<M_; i++) caffe_mul<complex<Dtype>>(K_, v_buffer, x_buffer+i*K_, m_buffer+i*K_);
  for(int i=0; i<M_*K_; i++)  m_buffer[i] *=  1/K_;
  
  //mat version
  for (int t=0; t<M_; t++){
    for (int i =0; i<K_; i++)
      R_vector[i] = (v+t*K_)[i];
    for (int i=0; i<K_; i++)
      for (int j=0; j<K_; j++){
	if (Trans)
	  (R_buffer+j*K_)[i] = R_vector[((i-j)+K_)%K_] * (i<j?-1.:1.);
	else
	  (R_buffer+i*K_)[j] = R_vector[((i-j)+K_)%K_] * (i<j?-1.:1.);
      }

    // M * K = (K * K * M)T
    if (IsVBatch)
      // K * 1 = (K * K * 1)
      caffe_cpu_gemv<Dtype>(CblasNoTrans, K_, K_, (Dtype)1., R_buffer, (x+t*K_), (Dtype)0., result+t*K_ ); 
    else{
      caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans, M_, K_, K_, (Dtype)1.,  x, R_buffer, (Dtype)0., result);
      break;
	}
    if (!IsVBatch) break;
  }
  
  for (int i=0; i<M_*K_; i++)
    if ( abs(10000* (std::real(m_buffer[i]) - result[i])) >1 )  LOG(INFO)<<"fft "<<std::real(m_buffer[i])<<" "<<std::imag(m_buffer[i])<<" mat "<<result[i];
  /*
  //LOG(INFO)<<"FFTSkewTest";
  //LOG(INFO)<<"v "<<v[0]<<" "<<v[1]<<" "<<v[2];
  //LOG(INFO)<<"x "<<x[0]<<" "<<x[1]<<" "<<x
  //if (Trans & ! IsVBatch){ K_=3; assist[1] = std::exp(std::complex<Dtype>(0,pi/3.)); assist[2]= assist[1] *assist[1]; M_=1; 
  //for (int i=0; i<K_; i++) LOG(INFO)<<"v"<<v[i]; for (int i=0; i<M_;i++) LOG(INFO)<<"x"<<(x+i*K_)[0]<<" "<<(x+i*K_)[1]<<" "<<(x+i*K_)[2];} 
  

  
  if (Trans & ! IsVBatch) LOG(INFO)<<"result "<<std::real(m_buffer[0])<<","<<std::imag(m_buffer[0])<<" ; "<<std::real(m_buffer[1])<<","<<std::imag(m_buffer[1])<<" ; "<<std::real(m_buffer[2])<<","<<std::imag(m_buffer[2]);


  for(int i=0; i<M_*K_;i++) result[i] = std::real(m_buffer[i])/K_;*/
}


template <typename Dtype>
void CirculantProjectionLayer<Dtype>::FillData(const int n, const int a, const int b, const Dtype* x, Dtype* y){

  if (a<=b){
    for(int i=0; i<n; i++){
      for(int j=0; j<a; j++) (y+ i*b)[j] =  (x+ i*a)[j];
      for(int j=a; j<b; j++) (y+ i*b)[j] = (Dtype)0;
    }
  }else{
     for(int i=0; i<n; i++)
       for(int j=0; j<b; j++) (y+ i*b)[j] =  (x+ i*a)[j];
  }


}


template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // static int i =0;

  LOG(INFO)<<"Backward";

  if (this->param_propagate_down_[0]) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_data();

    Dtype* data_buffer = this->data_buffer_.mutable_cpu_data();
    Dtype* data2_buffer = this->data_buffer_.mutable_cpu_diff();
    
    // Gradient with respect to weight
    // = skew_circ(Dsign * bottom_data)T * top_diff

    FillData(M_, N_, K_, top_diff, data2_buffer);

    // D * x
    for (int i =0; i<M_; i++)
      for (int j=0; j<K_; j++)
        (data_buffer+i*K_)[j] = (this->blobs_[2]->cpu_data()[j]>(Dtype)0.5?(bottom_data+i*K_)[j] : -(bottom_data+i*K_)[j]) / N_; //change


    this->FFTSkewCirc(1, 1, data_buffer, data2_buffer,  data_buffer);
    
    //sum
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1., data_buffer,
                         bias_multiplier_.cpu_data(), (Dtype)1.,
                         this->blobs_[0]->mutable_cpu_diff()); 
  }
  
  LOG(INFO)<<"Bias";
  if (bias_term_ && this->param_propagate_down_[1]){
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top[0]->cpu_diff(),
        bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }

    // Gradient with respect to bottom data top*R*D
  LOG(INFO)<<"BottomData";
  if (propagate_down[0]) {
    Dtype* data_buffer = this->data_buffer_.mutable_cpu_data();//M_*K_
    const Dtype* R_vector  = this->blobs_[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
  
    // D * circ(weight)T  * top_diff
    FillData(M_, N_, K_, top_diff, data_buffer);
    this->FFTSkewCirc(1, 0, R_vector, data_buffer, bottom[0]->mutable_cpu_diff());
    for(int i =0; i<M_; i++) for (int j=0; j<K_; j++) (bottom[0]->mutable_cpu_diff()+i*K_)[j] = (data_buffer+i*K_)[j] *( this->blobs_[2]->cpu_data()[j]>(Dtype)0.5? 1. : -1.);
  }

    LOG(INFO)<<"END_Back";
}

#ifdef CPU_ONLY
STUB_GPU(CirculantProjectionLayer);
#endif

INSTANTIATE_CLASS(CirculantProjectionLayer);
REGISTER_LAYER_CLASS(CirculantProjection);

} 
