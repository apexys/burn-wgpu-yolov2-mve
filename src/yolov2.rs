use burn::{module::Module, nn::{conv::{Conv2d, Conv2dConfig}, BatchNorm, ReLU, pool::{MaxPool2d, MaxPool2dConfig, AvgPool2d, AvgPool2dConfig}, BatchNormConfig}, tensor::{backend::Backend, Tensor, Shape}, config::Config};


//Implemented as per https://arxiv.org/abs/1612.08242
//with tensorflow conv blocks adapted from https://github.com/srihari-humbarwadi/YOLOv1-TensorFlow2.0/blob/master/yolo_v1.py
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend>{
    conv: Conv2d<B>,
    batch_norm: BatchNorm<B,2>,
    relu: ReLU,
    pool: Option<MaxPool2d>
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig{
    in_channels: usize,
    out_channels: usize,
    conv_size: usize,
    pool: bool
}

impl ConvBlockConfig{
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B>{
        ConvBlock{
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [self.conv_size, self.conv_size]).with_padding(burn::nn::PaddingConfig2d::Same).init(device),
            batch_norm: BatchNormConfig::new(self.out_channels).init(device),
            relu: ReLU::new(),
            pool: self.pool.then_some(MaxPool2dConfig::new([2,2]).with_strides([2,2]).with_padding(burn::nn::PaddingConfig2d::Valid).init())
        }
    }
}
 
impl<B: Backend> ConvBlock<B>{
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B,4>{
        let x = self.conv.forward(input);
        let x = self.batch_norm.forward(x);
        let x = self.relu.forward(x);
        let x = if let Some(pool) = self.pool.as_ref(){
            pool.forward(x)
        }else{
            x
        };
        x
    }
}


#[derive(Module, Debug)]
pub struct YOLOv2Model<B: Backend>{
    blocks: Vec<ConvBlock<B>>,
    output_conv: Conv2d<B>,
    output_avgpool: AvgPool2d
}

#[derive(Config, Debug)]
pub struct YOLOv2ModelConfig{
    input_size: [usize; 2],
    input_channels: usize,
    output_cells_per_axis: usize,
    output_classes: usize
}


impl YOLOv2ModelConfig{
    pub fn init<B: Backend>(&self, device: &B::Device) -> YOLOv2Model<B>{
        let blocks = vec![
            //Block 1
            ConvBlockConfig::new(1, 32, 3, true).init(device),
            //Block 2
            ConvBlockConfig::new(32, 64, 3, true).init(device),
            //Block 3
            ConvBlockConfig::new(64, 128, 3, false).init(device),
            ConvBlockConfig::new(128, 64, 1, false).init(device),
            ConvBlockConfig::new(64, 128, 3, true).init(device),
            //Block 4
            ConvBlockConfig::new(128, 256, 3, false).init(device),
            ConvBlockConfig::new(256, 128, 1, false).init(device),
            ConvBlockConfig::new(128, 256, 3, true).init(device),
            //Block 5
            ConvBlockConfig::new(256, 512, 3, false).init(device),
            ConvBlockConfig::new(512, 256, 1, false).init(device),
            ConvBlockConfig::new(256, 512, 3, false).init(device),
            ConvBlockConfig::new(512, 256, 1, false).init(device),
            ConvBlockConfig::new(256, 512, 3, true).init(device),
            //Block 6
            ConvBlockConfig::new(512, 1024, 3, false).init(device),
            ConvBlockConfig::new(1024, 512, 1, false).init(device),
            ConvBlockConfig::new(512, 1024, 3, false).init(device),
            ConvBlockConfig::new(1024, 512, 1, false).init(device),
            ConvBlockConfig::new(512, 1024, 3, false).init(device),
        ];
        //Forward a test tensor through all blocks to get the resulting shape
        let test_tensor: Tensor<B, 4> = Tensor::random(Shape::new([1,self.input_channels, self.input_size[0], self.input_size[1]]), burn::tensor::Distribution::Default, device);
        let result_tensor = blocks.iter().fold(test_tensor, |tensor, layer| {
            layer.forward(tensor)
        });
        let [_batchsize, last_channels, last_width, last_height] = result_tensor.shape().dims;
        let output_count = (self.output_cells_per_axis).pow(2) * (4 /*X,Y,W,H*/ + self.output_classes);
        let output_conv = Conv2dConfig::new([last_channels, output_count], [1,1]).init(device); //It's interesting that they do 1x1 convs here and not a 7x7 or whatever, since that would more accurately approach the linear layer from v1
        let output_avgpool = AvgPool2dConfig::new([last_width, last_height]).with_padding(burn::nn::PaddingConfig2d::Valid).with_strides([last_width, last_height]).init();
        YOLOv2Model{
            blocks,
            output_conv,
            output_avgpool
        }
    }
}

impl<B: Backend> YOLOv2Model<B>{
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B,2>{
        let x = self.blocks.iter().fold(input, |tensor, layer| layer.forward(tensor));
        let x = self.output_conv.forward(x);
        let x = self.output_avgpool.forward(x);
        let x: Tensor<B, 2> = x.squeeze::<3>(3).squeeze(2);
        x
    }
}
