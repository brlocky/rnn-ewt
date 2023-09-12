import pytest
from ..elliott.core.fibs import Fibonacci, FibonacciLevel


@pytest.fixture
def setup_fibs():
    fibs: Fibonacci = Fibonacci()

    # Return the object to be used in tests
    yield fibs


def run_test_fibonacci_retracement(setup_fibs, levels, pivot1, pivot2, expected_result):
    result = setup_fibs.get_retracements(pivot1, pivot2, levels)
    assert result == expected_result


def run_test_fibonacci_projection(setup_fibs, levels, pivot1, pivot2, pivot3, expected_result):
    result = setup_fibs.get_projections(pivot1, pivot2, pivot3, levels)
    assert result == expected_result


def test_retracement_1(setup_fibs):
    expected_result = {
        FibonacciLevel.FIB_0_236: 14292.0,
        FibonacciLevel.FIB_0_382: 13854.0,
        FibonacciLevel.FIB_0_5: 13500.0,
        FibonacciLevel.FIB_0_618: 13146.0,
        FibonacciLevel.FIB_0_786: 12642.0,
        FibonacciLevel.FIB_0_886: 12342.0
    }
    levels = expected_result.keys()
    pivot1, pivot2 = 12000, 15000
    run_test_fibonacci_retracement(setup_fibs, levels, pivot1, pivot2, expected_result)


def test_projection_1(setup_fibs):
    expected_result = {
        FibonacciLevel.FIB_0_5: 15390.0,
        FibonacciLevel.FIB_0_618: 15397.08,
        FibonacciLevel.FIB_0_786: 15407.16,
        FibonacciLevel.FIB_0_886: 15413.16,
        FibonacciLevel.FIB_1_0: 15420.0,
        FibonacciLevel.FIB_1_272: 15436.32,
        FibonacciLevel.FIB_1_414: 15444.84,
        FibonacciLevel.FIB_1_618: 15457.08,
        FibonacciLevel.FIB_2_0: 15480.0,
        FibonacciLevel.FIB_2_272: 15496.32,
        FibonacciLevel.FIB_2_414: 15504.84,
        FibonacciLevel.FIB_2_618: 15517.08,
        FibonacciLevel.FIB_3_618: 15577.08,
        FibonacciLevel.FIB_4_0: 15600,
        FibonacciLevel.FIB_4_236: 15614.16,
        FibonacciLevel.FIB_4_618: 15637.08
    }
    levels = expected_result.keys()
    pivot1, pivot2, pivot3 = 15336, 15396, 15360

    run_test_fibonacci_projection(
        setup_fibs, levels, pivot1, pivot2, pivot3, expected_result)
