"""Tokenizer conformance tests (spec 7.4, 7.5; fixtures 25 T1-T3)."""

from stylog.analysis.text import tokenize_text


def _kinds_texts(text: str) -> list[tuple[str, str]]:
    return [(token.kind, token.text) for token in tokenize_text(text)]


def test_t1_fixture_exact_tokens() -> None:
    text = "Don't re-enter—now. naïve café 3.14"
    assert _kinds_texts(text) == [
        ("word", "Don't"),
        ("word", "re"),
        ("word", "enter"),
        ("word", "now"),
        ("word", "naïve"),
        ("word", "café"),
        ("number", "3.14"),
    ]


def test_t1_offsets_are_codepoint_offsets() -> None:
    text = "Don't re-enter—now. naïve café 3.14"
    tokens = tokenize_text(text)
    for token in tokens:
        assert text[token.start : token.end] == token.text
    assert (tokens[0].start, tokens[0].end) == (0, 5)
    assert (tokens[-1].start, tokens[-1].end) == (31, 35)


def test_t2_fixture_exact_tokens() -> None:
    text = "'dogs' rock’n’roll x_y 1,000.50"
    assert _kinds_texts(text) == [
        ("word", "dogs"),
        ("word", "rock’n’roll"),
        ("word", "x"),
        ("word", "y"),
        ("number", "1,000.50"),
    ]


def test_t3_combining_mark_continues_word() -> None:
    text = "Café"
    tokens = tokenize_text(text)
    assert len(tokens) == 1
    assert tokens[0].kind == "word"
    assert tokens[0].text == "Café"
    assert len(tokens[0].text) == 5


def test_signs_never_part_of_number() -> None:
    assert _kinds_texts("-12 + 5") == [("number", "12"), ("number", "5")]


def test_apostrophe_requires_letter_before_and_after() -> None:
    assert _kinds_texts("dogs'") == [("word", "dogs")]
    assert _kinds_texts("'dogs'") == [("word", "dogs")]
    assert _kinds_texts("a''b") == [("word", "a"), ("word", "b")]
    assert _kinds_texts("it’") == [("word", "it")]


def test_apostrophe_retained_in_token_text() -> None:
    assert _kinds_texts("wouldn't") == [("word", "wouldn't")]
    assert _kinds_texts("l’homme") == [("word", "l’homme")]


def test_hyphens_and_dashes_terminate_words() -> None:
    assert _kinds_texts("re-enter") == [("word", "re"), ("word", "enter")]
    assert _kinds_texts("a–b—c") == [("word", "a"), ("word", "b"), ("word", "c")]


def test_mark_cannot_begin_word() -> None:
    assert _kinds_texts("́abc") == [("word", "abc")]


def test_number_separators_require_surrounding_digits() -> None:
    assert _kinds_texts("3.") == [("number", "3")]
    assert _kinds_texts("3..5") == [("number", "3"), ("number", "5")]
    assert _kinds_texts("1,2,,3") == [("number", "1,2"), ("number", "3")]
    assert _kinds_texts("1_000") == [("number", "1_000")]
    assert _kinds_texts(".5") == [("number", "5")]


def test_other_letter_numbers_are_not_number_starts() -> None:
    # U+2167 (Roman numeral eight, Nl) and U+00BD (vulgar fraction, No).
    assert _kinds_texts("Ⅷ ½") == []


def test_empty_and_token_free_text() -> None:
    assert tokenize_text("") == []
    assert tokenize_text("! ? … —") == []
