#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/layerop.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
    extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

    template <typename TypeParam>
    class LayerOpLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
    protected:
        LayerOpLayerTest()
            : blob_bottom_(new Blob<Dtype>(2, 7, 7, 2)),
              blob_top_(new Blob<Dtype>()) {
            // fill the values
            FillerParameter filler_param;
            UniformFiller<Dtype> filler(filler_param);
            filler.Fill(this->blob_bottom_);

            this->blob_bottom_vec_.push_back(this->blob_bottom_);
            blob_top_vec_.push_back(blob_top_);
        }
        virtual ~LayerOpLayerTest() {
            delete blob_bottom_;
            delete blob_top_;
        }
        Blob<Dtype>* const blob_bottom_;
        Blob<Dtype>* const blob_top_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
    };

    static constexpr char cmdline[] =
        "Hist /home/wenri/Git/Tamp/tools/gknlctr-01.txt 100";

    TYPED_TEST_CASE(LayerOpLayerTest, TestDtypesAndDevices);

    TYPED_TEST(LayerOpLayerTest, TestSetUp) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        LayerOpParameter* layer_op_param =
            layer_param.mutable_layer_op_param();
        layer_op_param->set_num_output(0);
        layer_op_param->set_bias_term(false);
        layer_op_param->set_cmdline(cmdline);
        shared_ptr<LayerOpLayer<Dtype> > layer
            (new LayerOpLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    }

    TYPED_TEST(LayerOpLayerTest, TestForward) {
        typedef typename TypeParam::Dtype Dtype;
        bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
        IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
        if (Caffe::mode() == Caffe::GPU &&
            sizeof(Dtype) == 4 && IS_VALID_CUDA) {
            LayerParameter layer_param;
            LayerOpParameter* layer_op_param =
                layer_param.mutable_layer_op_param();
            layer_op_param->set_num_output(0);
            layer_op_param->set_bias_term(false);
            layer_op_param->set_cmdline(cmdline);
            shared_ptr<LayerOpLayer<Dtype> > layer
                (new LayerOpLayer<Dtype>(layer_param));
            layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
            layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
            const Dtype* data = this->blob_top_->cpu_data();
            const int count = this->blob_top_->count();
            std::cout<<"Top Data("<<count<<"): ";
            for(int i=0; i<count; i++)
                std::cout<<data[i]<<" ";
            std::cout<<std::endl;

        } else {
            LOG(ERROR) << "Skipping test due to old architecture.";
        }
    }

    TYPED_TEST(LayerOpLayerTest, TestGradient) {
        typedef typename TypeParam::Dtype Dtype;
        bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
        IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
        if (Caffe::mode() == Caffe::GPU &&
            sizeof(Dtype) == 4 && IS_VALID_CUDA) {
            LayerParameter layer_param;
            LayerOpParameter* layer_op_param =
                layer_param.mutable_layer_op_param();
            layer_op_param->set_num_output(0);
            layer_op_param->set_bias_term(false);
            layer_op_param->set_cmdline(cmdline);
            LayerOpLayer<Dtype> layer(layer_param);
            GradientChecker<Dtype> checker(1e-2, 1e-3);
            checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                            this->blob_top_vec_);
        } else {
            LOG(ERROR) << "Skipping test due to old architecture.";
        }
    }

}  // namespace caffe
