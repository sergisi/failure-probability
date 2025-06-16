import math
from fractions import Fraction
from math import factorial as fac
from math import log, ceil, erf, sqrt
import dataclasses as dto
import functools as fun
import typer


# NOTE: Typer is not really needed, but I like it to build my own cli
app = typer.Typer()

@dto.dataclass
class Distribution:
    input_law: dict[int, float]

    def __add__(self, other: "Distribution") -> "Distribution":
        result = {}
        for a in self.input_law:
            for b in other.input_law:
                c = a + b
                result[c] = result.get(c, 0) + self.input_law[a] * other.input_law[b]
        return Distribution(result)

    def __mul__(self, other: "Distribution") -> "Distribution":
        result = {}
        for a in self.input_law:
            for b in other.input_law:
                c = a * b
                result[c] = result.get(c, 0) + self.input_law[a] * other.input_law[b]
        return Distribution(result)

    def clean_dist(self) -> "Distribution":
        result = {}
        for x, y in self.input_law.items():
            if y > 2 ** (-300):
                result[x] = y
        return Distribution(result)

    def for_coefficints(self, number_coefficients: int) -> "Distribution":
        distribution = Distribution({0: 1.0})
        i_bin = bin(number_coefficients)[2:]
        for ch in i_bin:
            distribution = (distribution + distribution).clean_dist()
            if ch == "1":
                distribution = (distribution + self).clean_dist()
        return distribution

    def tail_probability(self, t: int) -> float:
        s = 0
        ma = max(self.input_law.keys())
        if t >= ma:
            return 0
        for i in reversed(range(int(ceil(t)), ma)):
            s += self.input_law.get(i, 0) + self.input_law.get(-i, 0)
        return s


def _binomial(x: int, y: int) -> int:
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def _centered_binomial_pdf(k: int, x: int) -> float:
    return _binomial(2 * k, x + k) / 2.0 ** (2 * k)


def build_centered_binomial_law(k: int) -> Distribution:
    distribution = {}
    for i in range(-k, k + 1):
        distribution[i] = _centered_binomial_pdf(k, i)
    return Distribution(distribution)


@fun.lru_cache()
def compute_distribution(degree: int = 1024, size: int = 2) -> Distribution:
    e1 = build_centered_binomial_law(2)
    e2 = build_centered_binomial_law(2)  # Vector
    b = build_centered_binomial_law(2)  # Vector
    be = b * e2
    be_coeff = be.for_coefficints(degree * size)
    e1_coeff = e1.for_coefficints(degree)
    return e1_coeff + be_coeff


def _tail(degree: int = 1024, size: int = 2, target: int = 300) -> Fraction:
    res = compute_distribution(degree, size).tail_probability(target)
    return Fraction(res)


def bisect(f, a, b):
    fa = f(a)
    fb = f(b)
    if fa > fb:
        a, b = b, a
        fa, fb = fb, fa
    assert fa.numerator <= 0 and fb.numerator >= 0
    for _ in range(128):
        mid = (a + b) // 2
        fmid = f(mid)
        if fmid < 0:
            a = mid
        else:
            b = mid
    return mid, fmid


def _prob(degree: int = 1024, size: int = 2, target: int = 300) -> Fraction:
    """

    """
    return 1 - (1  - _tail(degree=degree, size=size, target=target)) ** 256


@app.command()
def probability(degree: int = 1024, size: int = 2, target: int = 300):
    print(_prob(degree, size, target))


@app.command()
def search_coefficient(prob_target: float = 2 ** (-128)) -> None:
    target = Fraction(prob_target) 
    coef, err = bisect(
        f=lambda x: _prob(target=x) - target,
        a=1000,
        b=1,
    )
    print(f"Coeff := {coef}, actual error := {float(err)}")


if __name__ == "__main__":
    app()


