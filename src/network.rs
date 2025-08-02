use crate::stock::Data;
use ndarray::{Array1, Array2};
use rand::{rng, seq::SliceRandom};
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::io::{self, Write};
use std::time::Duration;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU(f32), // Stores the alpha parameter
    Swish,
    Mish,
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    pub input_size: usize,
    pub layers: Vec<Layer>,
    pub activation: Activation,
}

impl Network {
    pub fn new(input_size: usize, layer_sizes: Vec<usize>, activation: Activation) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        // He initialization for ReLU family, Xavier for others
        let init_std = match activation {
            Activation::Sigmoid => (1.0 / prev_size as f32).sqrt(),
            _ => (2.0 / prev_size as f32).sqrt(), // He initialization
        };

        for &size in &layer_sizes {
            layers.push(Layer::new(size, prev_size, init_std));
            prev_size = size;
        }

        Self {
            input_size,
            layers,
            activation,
        }
    }

    pub fn feed_forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut activation = input.clone();
        for layer in self.layers.iter() {
            let z = &layer.weights.dot(&activation) + &layer.bias;
            activation = self.activate(&z);
        }
        activation
    }

    fn activate(&self, z: &Array1<f32>) -> Array1<f32> {
        match self.activation {
            Activation::Sigmoid => sigmoid(z),
            Activation::ReLU => relu(z),
            Activation::LeakyReLU(alpha) => leaky_relu(z, alpha),
            Activation::Swish => swish(z),
            Activation::Mish => mish(z),
        }
    }

    fn activate_prime(&self, z: &Array1<f32>) -> Array1<f32> {
        match self.activation {
            Activation::Sigmoid => sigmoid_prime(z),
            Activation::ReLU => relu_prime(z),
            Activation::LeakyReLU(alpha) => leaky_relu_prime(z, alpha),
            Activation::Swish => swish_prime(z),
            Activation::Mish => mish_prime(z),
        }
    }

    pub fn sgd(&mut self, eta: f32, epochs: usize, mini_batch_size: u32, training_data: Data) {
        let mut rng = rng();
        let mut training_pairs: Vec<(Array1<f32>, f32)> = Vec::new();

        // Collect training pairs from all stocks
        for stock_data in &training_data.data {
            let days = stock_data.training_input.len();
            // Ensure we have enough days to form at least one input
            if days >= self.input_size {
                // Slide a window of size input_size to create input-target pairs
                for i in 0..=(days - self.input_size) {
                    let input: Vec<f32> = stock_data.training_input[i..(i + self.input_size)]
                        .iter()
                        .map(|entry| entry.close)
                        .collect();
                    let target = if i + self.input_size < days {
                        stock_data.training_input[i + self.input_size].close
                    } else {
                        stock_data.real_value
                    };
                    training_pairs.push((Array1::from(input), target));
                }
            }
        }

        println!("\x1b[1;33m╔═════════════════════════════════════════════╗\x1b[0m");
        println!(
            "\x1b[1;33m║\x1b[1;34m         Training: SGD with Parameters       \x1b[1;33m║\x1b[0m"
        );
        println!(
            "\x1b[1;33m║\x1b[1;34m   η = {eta:.4} | batch = {mini_batch_size:>3} | epochs = {epochs:>3}   \x1b[1;33m║\x1b[0m"
        );
        println!("\x1b[1;33m╚═════════════════════════════════════════════╝\x1b[0m");
        for i in 0..epochs {
            let start = std::time::Instant::now();
            training_pairs.shuffle(&mut rng);
            for chunk in training_pairs.chunks(mini_batch_size as usize) {
                let mut nabla_w: Vec<Array2<f32>> = self
                    .layers
                    .iter()
                    .map(|layer| Array2::zeros((layer.layer_size, layer.weights.ncols())))
                    .collect();
                let mut nabla_b: Vec<Array1<f32>> = self
                    .layers
                    .iter()
                    .map(|layer| Array1::zeros(layer.layer_size))
                    .collect();

                // Calculate gradients
                for (input, target) in chunk {
                    let (delta_nabla_w, delta_nabla_b) = self.backprop(input, *target);
                    for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                        *nb += dnb;
                    }

                    for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                        *nw += dnw;
                    }
                }

                // Update newtork weights and bias
                for (layer, (nw, nb)) in self
                    .layers
                    .iter_mut()
                    .zip(nabla_w.iter().zip(nabla_b.iter()))
                {
                    layer.weights -= &(eta / mini_batch_size as f32 * nw);
                    layer.bias -= &(eta / mini_batch_size as f32 * nb);
                }
            }
            println!("\x1b[1;32mTrained epoch: [{i}|{epochs}]");
            let estimated = (epochs as u64 - i as u64) * start.elapsed().as_secs();
            println!(
                "\x1b[1;32mEstimated time: {:?}",
                Duration::from_secs(estimated)
            );
            if i % 100 == 0 {
                let path = format!("data/networks/checkpoint{i}");
                self.save_to_file(&path).unwrap();
            }
        }
    }

    fn backprop(&self, input: &Array1<f32>, target: f32) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut nabla_w: Vec<Array2<f32>> = self
            .layers
            .iter()
            .map(|layer| Array2::zeros((layer.layer_size, layer.weights.ncols())))
            .collect();
        let mut nabla_b: Vec<Array1<f32>> = self
            .layers
            .iter()
            .map(|layer| Array1::zeros(layer.layer_size))
            .collect();

        let mut activations = vec![input.clone()];
        let mut zs = Vec::new();
        for layer in &self.layers {
            let z = layer.weights.dot(activations.last().unwrap()) + &layer.bias;
            zs.push(z.clone());
            activations.push(self.activate(&z));
        }

        let output = activations.last().unwrap();
        let delta = output - target; // Cost function derivative
        let last_z = zs.last().unwrap();
        let delta = delta * self.activate_prime(last_z);

        nabla_b[self.layers.len() - 1] = delta.clone();
        nabla_w[self.layers.len() - 1] = delta.clone().to_shape((delta.len(), 1)).unwrap().dot(
            &activations[activations.len() - 2]
                .clone()
                .to_shape((1, activations[activations.len() - 2].len()))
                .unwrap(),
        );

        for l in (0..self.layers.len() - 1).rev() {
            let z = &zs[l];
            let sp = self.activate_prime(z);
            let delta_next = &self.layers[l + 1].weights.t().dot(&delta);
            let delta = delta_next * sp;

            nabla_b[l] = delta.clone();
            nabla_w[l] = delta.clone().to_shape((delta.len(), 1)).unwrap().dot(
                &activations[l]
                    .clone()
                    .to_shape((1, activations[l].len()))
                    .unwrap(),
            );
        }

        (nabla_w, nabla_b)
    }

    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let encoded = bincode::serialize(self).expect("Serialization failed");
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let decoded: Self = bincode::deserialize(&buffer).expect("Deserialization failed");
        Ok(decoded)
    }
}

pub fn sigmoid(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| 1.0 / (1.0 + (-z_i).exp()))
}

pub fn sigmoid_prime(z: &Array1<f32>) -> Array1<f32> {
    let s = sigmoid(z);
    &s * (1.0 - &s)
}

pub fn relu(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| z_i.max(0.0))
}

pub fn relu_prime(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| if z_i > 0.0 { 1.0 } else { 0.0 })
}

pub fn leaky_relu(z: &Array1<f32>, alpha: f32) -> Array1<f32> {
    z.mapv(|z_i| if z_i > 0.0 { z_i } else { alpha * z_i })
}

pub fn leaky_relu_prime(z: &Array1<f32>, alpha: f32) -> Array1<f32> {
    z.mapv(|z_i| if z_i > 0.0 { 1.0 } else { alpha })
}

pub fn swish(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| z_i / (1.0 + (-z_i).exp()))
}

pub fn swish_prime(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| {
        let sig = 1.0 / (1.0 + (-z_i).exp());
        sig + z_i * sig * (1.0 - sig)
    })
}

pub fn mish(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| z_i * (1.0 + (-z_i).exp()).ln_1p().tanh())
}

pub fn mish_prime(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| {
        let omega = 4.0 * (z_i + 1.0) + 4.0 * (2.0 * z_i).exp() + (3.0 * z_i).exp();
        let delta = 1.0 + (z_i + 1.0).powi(2) + 2.0 * z_i.exp();
        omega / delta.powi(2)
    })
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub layer_size: usize,
    pub weights: Array2<f32>, // shape: [layer_size, input_size]
    pub bias: Array1<f32>,    // shape: [layer_size]
}

impl Layer {
    pub fn new(layer_size: usize, input_size: usize, init_std: f32) -> Self {
        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, init_std).unwrap();

        let weights = Array2::from_shape_fn((layer_size, input_size), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(layer_size); // Initialize biases to 0

        Self {
            layer_size,
            weights,
            bias,
        }
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let last = self.layer_size - 1;
        writeln!(f, " Layer")?;

        for j in 0..self.layer_size {
            let (start, end) = match j {
                0 => ('┌', '┐'),
                i if i == last => ('└', '┘'),
                _ => ('│', '│'),
            };

            write!(f, " {start} ")?;
            for w in self.weights.row(j).iter() {
                write!(f, "{w:>7.3} ")?;
            }
            writeln!(f, "{end}  |  {start} {:>7.3} {end}", self.bias[j])?;
        }

        Ok(())
    }
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for layer in &self.layers {
            writeln!(f, "{layer}")?;
        }

        Ok(())
    }
}
