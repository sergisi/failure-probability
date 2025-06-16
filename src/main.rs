use clap::{Parser, Subcommand};
use once_cell::sync::OnceCell;
use rug::ops::*;
use rug::Complete;
use rug::Rational;
use std::collections::BTreeMap;

/// Helper: factorial n!
fn factorial(n: u32) -> rug::Integer {
    let mut f = rug::Integer::from(1);
    for i in 1..=n {
        f *= i;
    }
    f
}

/// Binomial coefficient: C(x, y), returns 0 if invalid
fn binomial(x: i32, y: i32) -> rug::Integer {
    if y < 0 || y > x {
        return rug::Integer::from(0);
    }
    let xf = factorial(x as u32);
    let yf = factorial(y as u32);
    let x_minus_yf = factorial((x - y) as u32);
    xf / (yf * x_minus_yf)
}

/// Centered binomial PDF: C(2k, x+k) / 2^(2k)
fn centered_binomial_pdf(k: i32, x: i32) -> Rational {
    let num = binomial(2 * k, x + k);
    let denom = rug::Integer::from(1) << (2 * k);
    Rational::from((num, denom))
}

/// Distribution struct holding probabilities via a BTreeMap for sorted keys
#[derive(Clone, Debug)]
struct Distribution {
    ppmf: BTreeMap<i32, Rational>,
}

impl Distribution {
    fn new() -> Self {
        Self {
            ppmf: BTreeMap::new(),
        }
    }

    fn from_map(map: BTreeMap<i32, Rational>) -> Self {
        Self { ppmf: map }
    }

    /// Convolution: self + other
    fn add(&self, other: &Self) -> Self {
        let mut result: BTreeMap<i32, Rational> = BTreeMap::new();
        for (&a, pa) in &self.ppmf {
            for (&b, pb) in &other.ppmf {
                let c = a + b;
                let entry = result.get(&c).unwrap_or(Rational::ZERO);
                result.insert(c, (entry + &(pa * pb).complete()).complete());
            }
        }
        Distribution::from_map(result)
    }

    /// Multiplication-style convolution: self * other
    fn mul(&self, other: &Self) -> Self {
        let mut result = BTreeMap::new();
        for (&a, pa) in &self.ppmf {
            for (&b, pb) in &other.ppmf {
                let c = a * b;
                let entry = result.get(&c).unwrap_or(Rational::ZERO);
                result.insert(c, (entry + &(pa * pb).complete()).complete());
            }
        }
        Distribution::from_map(result)
    }

    /// Remove probabilities < 2^-300
    fn clean(&self) -> Self {
        let thresh = Rational::from((rug::Integer::from(1), rug::Integer::from(1) << 300));
        Distribution::from_map(
            self.ppmf
                .iter()
                .filter(|(_, p)| *p > &thresh)
                .map(|(&x, p)| (x, p.clone()))
                .collect(),
        )
    }

    /// Exponentiation via binary exponentiation for coefficient convolution
    fn for_coefficients(&self, n: u32) -> Self {
        let mut res = Distribution::from_map({
            let mut m = BTreeMap::new();
            m.insert(0, Rational::from(1));
            m
        });
        let base = self.clone();
        for bit in (0..32).rev().map(|i| ((n >> i) & 1) != 0) {
            res = res.add(&res).clean();
            if bit {
                res = res.add(&base).clean();
            }
        }
        res
    }

    /// Tail probability sum for i >= ⌈t⌉
    fn tail_probability(&self, t: i32) -> Rational {
        let ma = *self.ppmf.keys().max().unwrap_or(&0);
        if t >= ma {
            return Rational::from(0);
        }
        let mut sum = Rational::from(0);
        for i in (t..=ma).rev() {
            sum += self
                .ppmf
                .get(&i)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
        }
        sum
    }

    /// Tail probability sum for |i| >= ⌈t⌉
    fn tail_probability_abs(&self, t: i32) -> Rational {
        let ma = *self.ppmf.keys().max().unwrap_or(&0);
        if t >= ma {
            return Rational::from(0);
        }
        let mut sum = Rational::from(0);
        for i in (t..=ma).rev() {
            sum += self
                .ppmf
                .get(&i)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
            sum += self
                .ppmf
                .get(&-i)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
        }
        sum
    }
}

/// Build centered binomial law for k
fn build_centered_binomial_law(k: i32) -> Distribution {
    let mut m = BTreeMap::new();
    for i in -k..=k {
        m.insert(i, centered_binomial_pdf(k, i));
    }
    Distribution::from_map(m)
}

/// Compute distribution cache
fn compute_distribution(degree: u32, size: u32) -> Distribution {
    static CACHE: OnceCell<Distribution> = OnceCell::new();
    CACHE
        .get_or_init(|| {
            let e1 = build_centered_binomial_law(2);
            let e2 = build_centered_binomial_law(2);
            let b = build_centered_binomial_law(2);
            let be = b.mul(&e2);
            let be_coeff = be.for_coefficients(degree * size);
            let e1_coeff = e1.for_coefficients(degree);
            e1_coeff.add(&be_coeff)
        })
        .clone()
}

/// Tail fraction
fn tail(degree: u32, size: u32, target: i32) -> Rational {
    let dist = compute_distribution(degree, size);
    dist.tail_probability(target)
}

/// Probability calculation
fn prob(degree: u32, size: u32, target: i32) -> Rational {
    let t = tail(degree, size, target);
    // Compute 1 - (1 - t)^256
    let one = Rational::from(1);
    let inner = &one - t;
    one - inner.pow(256)
}

/// CLI definition
#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
    #[clap(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Probability {
        #[clap(long, default_value = "1024")]
        degree: u32,
        #[clap(long, default_value = "2")]
        size: u32,
        #[clap(long, default_value = "300")]
        target: i32,
    },
    SearchCoefficient {
        #[clap(long, default_value = "2.938735877055719e-39")] // ≈2^(-128)
        prob_target: f64,
    },
    Check,
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Probability {
            degree,
            size,
            target,
        } => {
            let p = prob(degree, size, target);
            println!("{}", p.to_f64());
        }
        Commands::SearchCoefficient { prob_target } => {
            let target = Rational::from_f64(prob_target)
                .expect("Prob target should be convertible to rational");
            // bisection search between [1000,1]
            let mut a = 2000;
            let mut b = 1;
            let mut fa = prob(1024, 2, a); // adjust signature as needed
            let mut fb = prob(1024, 2, b);
            if fa > fb {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
            assert!(
                fa <= target && target <= fb,
                "f(a) = {}; f(b) = {}",
                fa.to_f64(),
                fb.to_f64()
            );
            let mut mid = a;
            for _ in 0..128 {
                mid = (a + b) / 2;
                let fmid = prob(1024, 2, mid);
                if fmid < target {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            println!(
                "Coeff := {}, actual error := {}",
                mid,
                prob(1024, 2, mid).to_f64()
            );
        }
        Commands::Check => {
            let dist = compute_distribution(1024, 2);
            let ma = *dist.ppmf.keys().max().unwrap();
            for i in 0..=ma {
                if dist.ppmf.get(&i) != dist.ppmf.get(&-i) {
                    panic!(
                        "Asymmetry detected at ±{}: {:?} != {:?}",
                        i,
                        dist.ppmf.get(&i),
                        dist.ppmf.get(&-i)
                    );
                }
            }
            println!("Distribution is symmetric up to computed support.");
        }
    }
}
