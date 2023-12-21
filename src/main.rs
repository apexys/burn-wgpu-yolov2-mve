use std::time::Instant;

use burn::{backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi}, nn::loss::MSELoss, optim::{AdamConfig, GradientsParams, Optimizer}, module::Module, tensor::{Tensor, Shape, backend::Backend}};

use crate::yolov2::{YOLOv2Model, YOLOv2ModelConfig};

mod yolov2;

type SelectedBackend = Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>;

fn main() {
    let device: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::default();
            let mut model: YOLOv2Model<SelectedBackend> = YOLOv2ModelConfig::new([256,256], 1, 16, 1).init(&device);
            println!("Model parameter count: {}", model.num_params());
            let loss: MSELoss<SelectedBackend> = MSELoss::new();
            let mut optimizer = AdamConfig::new().init();
            let batch_sizes_to_test = [1,4,8];
            let runs_per_batch = 4;
            let learning_rate = 1.0 / runs_per_batch as f64;
            
            for b in batch_sizes_to_test{
                println!("Testing batch size {b}");
                let batch_input: Tensor<SelectedBackend, 4> = Tensor::random(Shape::new([b,1, 256,256]), burn::tensor::Distribution::Default, &device);
                let batch_output: Tensor<SelectedBackend, 2> = Tensor::random(Shape::new([b, 16usize.pow(2) * 5]), burn::tensor::Distribution::Default, &device);
                println!("Batch output shape: {:?}", batch_output.shape());
                let mut time_us = 0;
                for i in 0 ..= runs_per_batch{
                    let now = Instant::now();
                    let predictions = model.forward(batch_input.clone());
                    let loss = loss.forward(predictions.clone(), batch_output.clone(), burn::nn::loss::Reduction::Auto);
                    let grads = loss.backward();
                    let grad_params = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(learning_rate, model, grad_params);
                    SelectedBackend::sync(&device);
                    if i > 0{
                        time_us += now.elapsed().as_micros();
                        println!("Epoch {} took {:.3}ms and reached loss {:.3}",i , now.elapsed().as_secs_f32() * 1000.0, loss.into_scalar());
                    }else{
                        println!("Warmup epoch took {}ms", now.elapsed().as_secs_f32() * 1000.0);
                    }
                }
                time_us /= runs_per_batch;
                println!("Average time per batch at batch_size={b}: {:.3}ms", time_us as f64 / 1000.0);
            }    
}
