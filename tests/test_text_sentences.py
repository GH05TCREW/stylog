"""Sentence segmentation conformance tests (spec 7.8; fixture 25)."""

from stylog.analysis.text import segment_sentences


def test_sentence_fixture_four_sentences() -> None:
    text = "Dr. Smith left. Value 3.14! Really?"
    assert segment_sentences(text) == ["Dr.", "Smith left.", "Value 3.14!", "Really?"]


def test_no_abbreviation_dictionary() -> None:
    # A smart abbreviation list would keep "Dr. Smith" together: that is a failure.
    assert segment_sentences("Dr. Smith") == ["Dr.", "Smith"]


def test_terminal_cluster_and_closers() -> None:
    text = "“What?!” She asked"
    assert segment_sentences(text) == ["“What?!”", "She asked"]


def test_ascii_quote_and_bracket_closers() -> None:
    assert segment_sentences('She said "hi." Done.') == ['She said "hi."', "Done."]
    assert segment_sentences("(Really?!)] Next.") == ["(Really?!)]", "Next."]


def test_decimal_point_is_not_terminal() -> None:
    assert segment_sentences("Value 3.14!") == ["Value 3.14!"]
    assert segment_sentences("Pi is 3.14.") == ["Pi is 3.14."]


def test_trailing_decimal_point_at_paragraph_end_is_terminal() -> None:
    assert segment_sentences("It ends 5.") == ["It ends 5."]


def test_residual_without_terminal_forms_sentence() -> None:
    assert segment_sentences("no terminal here") == ["no terminal here"]


def test_boundary_requires_whitespace_or_end() -> None:
    assert segment_sentences("a.b!c") == ["a.b!c"]
    assert segment_sentences("one.Two.") == ["one.Two."]


def test_edge_whitespace_trimmed_from_spans() -> None:
    assert segment_sentences("  Hello…  World.  ") == ["Hello…", "World."]


def test_all_terminal_code_points() -> None:
    text = "A。 B！ C？ D… E! F? G."
    assert segment_sentences(text) == ["A。", "B！", "C？", "D…", "E!", "F?", "G."]


def test_empty_and_whitespace_only() -> None:
    assert segment_sentences("") == []
    assert segment_sentences("   ") == []


def test_multiple_terminals_collapse_into_one_cluster() -> None:
    assert segment_sentences("Wait... Really?! Done") == ["Wait...", "Really?!", "Done"]
