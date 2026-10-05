import pytest

from calc.metrics import (
    CalcError, cagr, gross_margin, growth_rate, net_margin, operating_margin, ratio,
)


def test_gross_margin():
    assert gross_margin(revenue=4235102, cogs=2100000) == pytest.approx(0.504144, rel=1e-5)


def test_operating_margin():
    assert operating_margin(operating_income=812300, revenue=4235102) == pytest.approx(0.19180, rel=1e-4)


def test_net_margin():
    assert net_margin(net_income=500000, revenue=4235102) == pytest.approx(0.11807, rel=1e-4)


def test_growth_rate_positive():
    assert growth_rate(current=1500, previous=1200) == pytest.approx(0.25)


def test_growth_rate_negative():
    assert growth_rate(current=900, previous=1200) == pytest.approx(-0.25)


def test_cagr():
    # Doubling over 3 years -> ~25.99% CAGR
    assert cagr(begin_value=100, end_value=200, periods=3) == pytest.approx(0.2599, rel=1e-3)


def test_cagr_rejects_negative_values():
    with pytest.raises(CalcError):
        cagr(begin_value=-100, end_value=200, periods=3)


def test_ratio():
    assert ratio(numerator=50, denominator=200) == 0.25


@pytest.mark.parametrize("fn, kwargs", [
    (gross_margin, {"revenue": 0, "cogs": 100}),
    (operating_margin, {"operating_income": 100, "revenue": 0}),
    (net_margin, {"net_income": 100, "revenue": 0}),
    (growth_rate, {"current": 100, "previous": 0}),
    (cagr, {"begin_value": 0, "end_value": 100, "periods": 3}),
    (ratio, {"numerator": 100, "denominator": 0}),
])
def test_division_by_zero_raises_calc_error_not_crash(fn, kwargs):
    with pytest.raises(CalcError):
        fn(**kwargs)
