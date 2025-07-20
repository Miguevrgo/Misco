use crate::stock::Data;
use ndarray::{Array1, Array2};
use rand::rng;
use rand_distr::Distribution;

pub struct Network {
    pub input_size: usize,
    pub num_layers: usize,
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(input_size: usize, num_layers: usize) -> Self {
        Self {
            input_size,
            num_layers,
            layers: Vec::with_capacity(num_layers),
        }
    }

    pub fn feed_forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut activation = input.clone();
        for layer in &self.layers {
            let z = layer.weights.dot(&activation) + &layer.bias;
            activation = sigmoid(&z);
        }
        activation
    }

    pub fn SGD(&mut self, eta: f32, epochs: usize, mini_batch_size: u32, training_data: Data) {
        unimplemented!()
    }
}

pub fn sigmoid(z: &Array1<f32>) -> Array1<f32> {
    z.mapv(|z_i| 1.0 / (1.0 + (-z_i).exp()))
}

pub fn sigmoid_prime(z: &Array1<f32>) -> Array1<f32> {
    sigmoid(z).mapv(|s| s * (1.0 - s))
}

pub struct Layer {
    pub layer_size: usize,
    pub weights: Array2<f32>, // shape: [layer_size, input_size]
    pub bias: Array1<f32>,    // shape: [layer_size]
}

impl Layer {
    pub fn new(layer_size: usize, input_size: usize) -> Self {
        let mut rng = rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let weights =
            Array2::from_shape_fn((layer_size, input_size), |_| normal.sample(&mut rng) as f32);

        let bias = Array1::from_shape_fn(layer_size, |_| normal.sample(&mut rng) as f32);

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
