use crate::stock::Data;
use ndarray::{Array1, Array2};
use rand::{rng, seq::SliceRandom};
use rand_distr::Distribution;

pub struct Network {
    pub input_size: usize,
    pub num_layers: usize,
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(input_size: usize, num_layers: usize) -> Self {
        let layers = Vec::with_capacity(num_layers);
        Self {
            input_size,
            num_layers,
            layers,
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
        let mut rng = rng();
        let mut training_pairs: Vec<(Array1<f32>, f32)> = training_data
            .data
            .iter()
            .map(|stock_data| {
                let input: Array1<f32> = stock_data
                    .training_input
                    .iter()
                    .flat_map(|entry| vec![entry.open, entry.high, entry.low, entry.close])
                    .collect();
                (input, stock_data.real_value)
            })
            .collect();

        for _ in 0..epochs {
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
            activations.push(sigmoid(&z));
        }

        let output = activations.last().unwrap();
        let delta = output - target; // Cost function derivative
        let last_z = zs.last().unwrap();
        let delta = delta * sigmoid_prime(last_z);

        nabla_b[self.num_layers - 1] = delta.clone();
        nabla_w[self.num_layers - 1] = delta.clone().to_shape((delta.len(), 1)).unwrap().dot(
            &activations[activations.len() - 2]
                .clone()
                .to_shape((1, activations[activations.len() - 2].len()))
                .unwrap(),
        );

        for l in (0..self.num_layers - 1).rev() {
            let z = &zs[l];
            let sp = sigmoid_prime(z);
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
