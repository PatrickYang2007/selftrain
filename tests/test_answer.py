import pytest

from src.answer import (
    ANSWER_LINE,
    BOXED,
    LAST_NUMBER,
    canonical,
    extract_answer,
    is_correct,
    parse_gsm8k_answer,
)


@pytest.mark.parametrize(
    "response, value",
    [
        ("Step 1...\nAnswer: 18", 18),
        ("Answer: 1,234", 1234),
        ("Answer: 1,234,567.5", 1234567.5),
        ("Answer: $18", 18),
        ("Answer: $1,000.00", 1000),
        ("Answer: 18.", 18),
        ("Answer: 2.5.", 2.5),
        ("Answer: -7", -7),
        ("Answer: -$5", -5),
        ("Answer: −7", -7),  # unicode minus
        ("Answer: .5", 0.5),
        ("Answer: 18 dollars", 18),
        ("Answer: 25%", 25),
        ("Answer: 3/4", 0.75),
        ("**Answer:** 42", 42),
        ("**Answer: 42**", 42),
        ("answer: 42", 42),
        ("Final Answer: 42", 42),
        ("Answer : 42", 42),
        ("Answer: 5 + 3 = 8", 8),
        ("Answer:\n\n18", 18),
    ],
)
def test_answer_line(response, value):
    ex = extract_answer(response)
    assert ex.method == ANSWER_LINE
    assert ex.value == pytest.approx(value)


@pytest.mark.parametrize(
    "response, value",
    [
        ("so the total is \\(\\boxed{128}\\) dollars for 16 glasses.", 128),
        ("\\[ \\boxed{1,000} \\]", 1000),
        ("\\boxed{-520}", -520),
        ("\\boxed{\\$18}", 18),
        ("\\boxed{18 \\text{ dollars}}", 18),
        ("\\boxed{\\frac{3}{4}}", 0.75),
        ("\\boxed{\\dfrac{1}{2}}", 0.5),
        ("\\boxed{x = 5}", 5),
        ("first \\boxed{3}, then corrected: \\boxed{7}", 7),
        ("\\boxed{12", 12),  # truncated before the closing brace
        ("Answer: \\boxed{42}", 42),
    ],
)
def test_boxed(response, value):
    ex = extract_answer(response)
    assert ex.method == BOXED
    assert ex.value == pytest.approx(value)


def test_answer_line_beats_earlier_boxed():
    assert extract_answer("\\boxed{3}\nAnswer: 4").value == 4


@pytest.mark.parametrize(
    "response, value",
    [
        ("**Final Answer:**\nIn 2010 the car is worth \\boxed{23238}", 23238),
        ("**State the final answer:**\nThe total for 500 computers is:\n\\boxed{385000}", 385000),
        ("- Answer: 2 hours\nThen the rest.\nTotal: \\boxed{7}", 7),
    ],
)
def test_later_boxed_beats_answer_line(response, value):
    ex = extract_answer(response)
    assert ex.method == BOXED
    assert ex.value == value


def test_latex_thousands_separator():
    assert extract_answer("\\boxed{1{,}000}").value == 1000


def test_boxed_without_number_falls_back():
    ex = extract_answer("It costs 9 dollars. \\boxed{nine}")
    assert ex.method == LAST_NUMBER
    assert ex.value == 9


def test_last_answer_line_wins():
    text = "Answer: 5\nWait, I made a mistake.\nAnswer: 7"
    assert extract_answer(text).value == 7


def test_numbers_after_answer_line_are_ignored():
    # The answer line is authoritative even if the model keeps talking.
    text = "Answer: 12\nThis took 3 steps."
    assert extract_answer(text).value == 12


def test_comma_separated_list_is_not_merged():
    assert extract_answer("Answer: 3, 4").value == 3


def test_missing_answer_line_falls_back_to_last_number():
    ex = extract_answer("She has 3 apples and buys 4 more, so 7 in total.")
    assert ex.method == LAST_NUMBER
    assert ex.value == 7


def test_answer_line_without_number_falls_back():
    ex = extract_answer("Total is 3 + 4 = 7.\nAnswer: seven")
    assert ex.method == LAST_NUMBER
    assert ex.value == 7


def test_fallback_handles_commas_and_decimals():
    assert extract_answer("The total cost is $1,250.50.").value == 1250.5


def test_no_number_fails():
    ex = extract_answer("I don't know.")
    assert ex.failed
    assert ex.value is None and ex.method is None


def test_empty_response_fails():
    assert extract_answer("").failed


def test_zero_denominator_falls_back():
    ex = extract_answer("Answer: 3/0")
    # 3/0 is not a valid fraction; fallback picks the last bare number.
    assert ex.method == LAST_NUMBER


@pytest.mark.parametrize(
    "solution, value",
    [
        ("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72", 72),
        ("...\n#### 1,000", 1000),
        ("...\n#### -3", -3),
        ("...\n#### 2.5", 2.5),
        ("...\n####  18 ", 18),
    ],
)
def test_parse_gsm8k(solution, value):
    assert parse_gsm8k_answer(solution) == value


def test_parse_gsm8k_requires_marker():
    with pytest.raises(ValueError):
        parse_gsm8k_answer("no marker here 72")


def test_is_correct():
    assert is_correct(18.0, 18)
    assert is_correct(18.0000001, 18)
    assert not is_correct(18.01, 18)
    assert not is_correct(None, 18)
    assert is_correct(-3.0, -3)
    assert is_correct(0.0, 0)


def test_canonical():
    assert canonical(18.0) == "18"
    assert canonical(18) == "18"
    assert canonical(2.50) == "2.5"
    assert canonical(-0.0) == "0"
    assert canonical(0.0) == "0"
    assert canonical(None) is None
    assert canonical(18.0000001) == canonical(18.0)
