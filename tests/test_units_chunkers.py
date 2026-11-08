import pytest

from chunkbench.chunkers import fixed, headings, parse, recursive
from chunkbench.units import chars, graphemes, words


def texts(text, spans):
    return [text[a:b] for a, b in spans]


def test_graphemes_keep_vowel_signs_and_conjuncts():
    assert texts("क्षमा", graphemes("क्षमा")) == ["क्ष", "मा"]
    assert texts("किताब", graphemes("किताब")) == ["कि", "ता", "ब"]
    assert texts("ab c", graphemes("ab c")) == ["a", "b", "c"]


def test_words_and_chars_skip_whitespace():
    assert texts("a  bc\nd", words("a  bc\nd")) == ["a", "bc", "d"]
    assert len(chars("a b")) == 2


def test_fixed_with_overlap():
    t = "a b c d e f g"
    assert texts(t, fixed(3, 1)(t)) == ["a b c", "c d e", "e f g"]
    with pytest.raises(ValueError):
        fixed(3, 3)


def test_recursive_prefers_paragraphs_then_sentences():
    t = "one two three.\n\nfour five six. seven eight."
    assert texts(t, recursive(3)(t)) == ["one two three.", "four five six.", "seven eight."]


def test_recursive_splits_on_danda():
    t = "पहला वाक्य यहाँ। दूसरा वाक्य यहाँ।"
    assert texts(t, recursive(3)(t)) == ["पहला वाक्य यहाँ।", "दूसरा वाक्य यहाँ।"]


def test_headings_keep_small_sections_whole():
    t = "# A\nx y\n\n# B\nz"
    assert texts(t, headings(10)(t)) == ["# A\nx y", "# B\nz"]


def test_parse():
    assert parse("fixed:10,2")[0] == "fixed:10,2"
    with pytest.raises(ValueError):
        parse("semantic:100")


@pytest.mark.parametrize(
    "spec", ["fixed:abc", "fixed:0", "recursive:0", "headings:-5", "fixed:10,-1", "recursive:50,10"]
)
def test_bad_strategy_specs_are_rejected(spec):
    with pytest.raises(ValueError, match="strategy"):
        parse(spec)
