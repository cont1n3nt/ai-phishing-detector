import pytest
from src.features import clean_text, validate_text, preprocess_texts


class TestCleanText:
    @pytest.mark.parametrize("raw, expected", [
        ("HELLO WORLD",              "hello world"),
        ("Hello!!! World??? #win",   "hello world win"),
        ("hello    world",           "hello world"),
        ("  spaced out  ",           "spaced out"),
        ("",                         ""),
        ("   ",                      ""),
    ])
    def test_transforms_text_correctly(self, raw, expected):
        assert clean_text(raw) == expected

    def test_replaces_urls_with_placeholder(self):
        result = clean_text("check https://evil.com and http://phish.ng now")
        assert "url" in result
        assert "http" not in result and "evil" not in result

    def test_handles_non_string_inputs(self):
        assert clean_text(123) == ""
        assert clean_text(None) == ""
        assert clean_text([]) == ""

    def test_handles_mixed_language_text(self):
        result = clean_text("Привет! Click here http://x.ru")
        assert "url" in result
        assert "привет" not in result  # only a-z, 0-9 preserved


class TestValidateText:
    @pytest.mark.parametrize("payload, expected_error", [
        (None,                        "No JSON received"),
        ({"foo": "bar"},              "No key 'text' in JSON"),
        ({"text": 123},               "Text must be a string"),
        ({"text": "a" * 5},           "Text too short"),
        ({"text": " \n\t "},          "Text too short"),
    ])
    def test_rejects_invalid_inputs(self, payload, expected_error):
        is_valid, msg = validate_text(payload)
        assert not is_valid
        assert msg.startswith(expected_error)

    def test_accepts_valid_text(self):
        text = "Valid email message with enough characters"
        is_valid, result = validate_text({"text": text})
        assert is_valid
        assert result == text


class TestPreprocessTexts:
    def test_processes_multiple_texts(self):
        inputs = ["HELLO", "WORLD!!!"]
        result = preprocess_texts(inputs)
        assert result == ["hello", "world"]