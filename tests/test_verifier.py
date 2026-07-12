from __future__ import annotations

from fractions import Fraction

import pytest

from hrm_particle.verifier import (
    ExactArithmeticVerifier,
    UnsafeArithmeticExpression,
    extract_answer_text,
    normalize_numeric,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.5", Fraction(1, 2)),
        ("1/2", Fraction(1, 2)),
        ("50%", Fraction(1, 2)),
        (r"\frac{6}{8}", Fraction(3, 4)),
        ("1,250", Fraction(1250)),
        ("(2 + 3) / 10", Fraction(1, 2)),
        ("1e-3", Fraction(1, 1000)),
        ("0.1234567890123456789", Fraction(1234567890123456789, 10**19)),
    ],
)
def test_exact_numeric_normalization(text, expected):
    assert normalize_numeric(text) == expected


def test_extracts_boxed_and_final_answer_forms():
    assert extract_answer_text(r"work 2+2 ... \boxed{\frac{4}{2}}") == r"\frac{4}{2}"
    assert extract_answer_text("reasoning\nFinal answer: -3/4") == "-3/4"


def test_exact_verifier_accepts_equivalent_fractions_and_rejects_wrong_answer():
    verifier = ExactArithmeticVerifier()
    assert verifier(r"Reasoning. Final answer: \frac{6}{8}", "3/4") == 1.0
    assert verifier("Final answer: 0.76", "3/4") == 0.0


@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').system('echo pwned')",
        "open('/tmp/pwned', 'w')",
        "(1).__class__.__mro__",
        "[x for x in range(10)]",
        "2 ** 1000000",
        "1 / 0",
    ],
)
def test_verifier_rejects_executable_or_unbounded_syntax(attack):
    with pytest.raises(UnsafeArithmeticExpression):
        normalize_numeric(attack)
    assert ExactArithmeticVerifier()(f"Final answer: {attack}", "1") == 0.0


def test_malformed_box_is_not_accepted():
    result = ExactArithmeticVerifier().verify(r"\boxed{1/2", "1/2")
    assert not result.correct
    assert result.error is not None
