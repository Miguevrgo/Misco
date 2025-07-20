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
}

pub struct Layer {
    pub layer_size: usize,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl Layer {
    pub fn new(layer_size: usize, input_size: usize) -> Self {
        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let weights = (0..layer_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| normal.sample(&mut rng) as f32)
                    .collect()
            })
            .collect();

        let bias = (0..layer_size)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();

        Self {
            layer_size,
            weights,
            bias,
        }
    }
}

use std::fmt;

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let last = self.weights.len() - 1;

        for (j, weights) in self.weights.iter().enumerate() {
            let border = if j == 0 {
                '['
            } else if j == last {
                ']'
            } else {
                '|'
            };

            write!(f, "n{:>2}: {} ", j, border)?;
            for w in weights {
                write!(f, "{:>7.3} ", w)?;
            }
            writeln!(f, "{} + {:>7.3}", border, self.bias[j])?;
        }

        Ok(())
    }
}
