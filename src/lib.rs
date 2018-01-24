extern crate ndarray;
extern crate ndarray_parallel;
extern crate num_traits;

use ndarray::{Array1, Array3, Zip};
use ndarray_parallel::prelude::*;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum InterpError {
    Range,
    NoneArray,
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InterpError::Range => write!(f, "xi is not bound by x"),
            InterpError::NoneArray => write!(f, "Returned none when looking for data in x"),
        }
    }
}

impl Error for InterpError {
    fn description(&self) -> &str {
        match *self {
            InterpError::Range => "out of bounds",
            InterpError::NoneArray => "option is none",
        }
    }
}

pub fn lerp(
    x: &Array1<f32>,
    y: &Array1<f32>,
    xi: &Array1<f32>,
) -> Result<Array1<f32>, InterpError> {
    // This check takes about 10% of the time.
    let xf = x.into_iter().next().ok_or(InterpError::NoneArray)?;
    let xl = x.into_iter().last().ok_or(InterpError::NoneArray)?;
    if xi.iter().any(|xi| xi < xf || xi > xl) {
        return Err(InterpError::Range);
    }
    let mut output = Array1::<f32>::zeros(xi.len());
    Zip::from(&mut output).and(xi).par_apply(|output, &xi| {
        // We know xi is in range since we just checked, so unrwap is fine here
        let xidx = x.windows(2)
            .into_iter()
            .position(|xw| xw[0] <= xi && xw[1] >= xi)
            .unwrap();
        let x0 = x[xidx];
        let x1 = x[xidx + 1];
        let y0 = y[xidx];
        let y1 = y[xidx + 1];
        *output = y0 + (xi - x0) * ((y1 - y0) / (x1 - x0));
    });
    Ok(output)
}

pub fn lerp_unchecked(x: &Array1<f32>, y: &Array1<f32>, xi: &Array1<f32>) -> Array1<f32> {
    let mut output = Array1::<f32>::zeros(xi.len());
    Zip::from(&mut output).and(xi).par_apply(|output, &xi| {
        // Note we don't check that xi is in x and could possibly panic here. So use with caution.
        let xidx = x.windows(2)
            .into_iter()
            .position(|xw| xw[0] <= xi && xw[1] >= xi)
            .unwrap();
        let x0 = x[xidx];
        let x1 = x[xidx + 1];
        let y0 = y[xidx];
        let y1 = y[xidx + 1];
        *output = y0 + (xi - x0) * ((y1 - y0) / (x1 - x0));
    });
    output
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use num_traits::float::Float;

    #[test]
    fn interp_l() {
        let x = Array::linspace(1., 10., 10);
        let y = Array::from_iter(x.into_iter().map(|x| x.sin()));
        let xi = Array::linspace(1., 10., 20);

        let yi = lerp(&x, &y, &xi).unwrap();
        assert_eq!(
            yi,
            Array::from_vec(vec![
                0.84147096,
                0.8735993,
                0.90572757,
                0.58585423,
                0.22198087,
                -0.18969359,
                -0.6150254,
                -0.8206304,
                -0.9163723,
                -0.7801062,
                -0.45823354,
                -0.08227807,
                0.36128092,
                0.70946646,
                0.8669055,
                0.9285964,
                0.655167,
                0.36179554,
                -0.09111279,
                -0.5440211,
            ])
        );
    }

    #[test]
    fn interp_l_unckecked() {
        let x = Array::linspace(1., 10., 10);
        let y = Array::from_iter(x.into_iter().map(|x| x.sin()));
        let xi = Array::linspace(1., 10., 20);

        let yi = lerp_unchecked(&x, &y, &xi);
        assert_eq!(
            yi,
            Array::from_vec(vec![
                0.84147096,
                0.8735993,
                0.90572757,
                0.58585423,
                0.22198087,
                -0.18969359,
                -0.6150254,
                -0.8206304,
                -0.9163723,
                -0.7801062,
                -0.45823354,
                -0.08227807,
                0.36128092,
                0.70946646,
                0.8669055,
                0.9285964,
                0.655167,
                0.36179554,
                -0.09111279,
                -0.5440211,
            ])
        );
    }
}
