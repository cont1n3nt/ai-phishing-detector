import pytest
from src.features import clean_text, validate_text, preprocess_texts


@pytest.mark.parametrize("raw, expected", [
    ("HELLO WORLD",              "hello world"),
    ("Hello!!! World??? #win",   "hello world win"),
    ("hello    world",           "hello world"),
    ("  spaced out  ",           "spaced out"),
    ("",                         ""),
    ("   ",                      ""),
])
def test_text_lowercased_and_cleaned(raw, expected):
    assert clean_text(raw) == expected


def test_urls_replaced_with_token():
    result = clean_text("check https://evil.com and http://phish.ng now")
    assert "url" in result
    assert "evil" not in result


def test_non_string_returns_empty():
    assert clean_text(123) == ""
    assert clean_text(None) == ""


def test_mixed_language_keeps_english_only():
    result = clean_text("Привет! Click here http://x.ru")
    assert "url" in result
    assert "привет" not in result


@pytest.mark.parametrize("payload, expected_in_error", [
    (None,                        "missing"),
    ({"foo": "bar"},              "missing"),
    ({"text": 123},               "must be a string"),
    ({"text": "a" * 5},           "too short"),
    ({"text": " \n\t "},          "too short"),
])
def test_validate_rejects_bad_input(payload, expected_in_error):
    is_valid, msg = validate_text(payload)
    assert not is_valid
    assert expected_in_error in msg


def test_validate_accepts_good_input():
    text = "This is a valid email with enough characters"
    is_valid, result = validate_text({"text": text})
    assert is_valid
    assert result == text


def test_preprocess_batch():
    result = preprocess_texts(["HELLO", "WORLD!!!"])
    assert result == ["hello", "world"]
