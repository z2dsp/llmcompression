import pytest

pytest.importorskip('mistune')

from llm_compression.reddit_corpus import clean_comment


@pytest.mark.parametrize('body, expected', [
    ('**bold**, __strong__, *italic*, _emphasis_ and ~~deleted~~.',
     'bold, strong, italic, emphasis and deleted.'),
    ('***nested*** and **bold _with italics_**', 'nested and bold with italics'),
    ('# Heading\n\n> Quote **here**\n\n- First\n- Second', 'Heading\nQuote here\nFirst\nSecond'),
    ('First\n\n\nSecond\n \t\nThird\r\n\r\nFourth', 'First\nSecond\nThird\nFourth'),
    ('Read [the guide](https://example.com/a_(b) "Title") now.', 'Read the guide now.'),
    ('A [reference][ref].\n\n[ref]: https://example.com', 'A reference.'),
    ('![a useful description](https://example.com/image.png)', 'a useful description'),
    ('Visit https://example.com/a_(b). Then www.example.org! Or ftp://example.com/file.',
     'Visit . Then ! Or .'),
    ('See (https://example.com/a_(b)).', 'See ().'),
    ('<https://example.com>\n\nwww.example.org', ''),
    ('<p>Hi <strong>there</strong><br>again &amp; again</p><!-- comment -->',
     'Hi there\nagain & again'),
    ('before <a href="https://example.com">the link</a> after', 'before the link after'),
    ('<script>bad()</script><style>.a { color: red; }</style>keep', 'keep'),
    ('```python\nx = 2 * 3\ny = "a<b>"\n```\n\nUse `snake_case`.',
     'x = 2 * 3\ny = "a<b>"\nUse snake_case.'),
    ('>!a secret!< and ^small and ^(two words)', 'a secret and small and two words'),
    ('- [x] Done\n- [ ] Later', 'Done\nLater'),
    ('snake_case, 2 * 3, x^2, café 🌊', 'snake_case, 2 * 3, x^2, café 🌊'),
    ('---\n\n***\n\n___', ''),
    ('', ''),
])
def test_plain_text_preprocessing(body, expected):
    assert clean_comment(body) == expected


def test_tables_keep_text_without_markdown_separators():
    result = clean_comment('| Name | Value |\n| --- | --- |\n| Test | **yes** |')
    assert result.split() == ['Name', 'Value', 'Test', 'yes']
    assert '|' not in result and '**' not in result and '\n\n' not in result
