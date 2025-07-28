from delphyne.scripts.load_configs import extract_config_block

#####
##### Testing `extract_config_block``
#####


def test_extracts_simple_config_block():
    doc = "# @config\n# foo: bar\n# baz: qux\n# @end\nother: value"
    result = extract_config_block(doc)
    assert result == "foo: bar\nbaz: qux"


def test_handles_blank_lines_and_comments_before_block():
    doc = "\n#\n# @config\n# foo: bar\n# @end"
    result = extract_config_block(doc)
    assert result == "foo: bar"


def test_returns_none_if_config_not_at_top():
    doc = "foo: bar\n# @config\n# baz: qux\n# @end"
    result = extract_config_block(doc)
    assert result is None


def test_returns_none_if_end_missing():
    doc = "# @config\n# foo: bar"
    result = extract_config_block(doc)
    assert result is None


def test_handles_empty_config_block():
    doc = "# @config\n# @end"
    result = extract_config_block(doc)
    assert result == ""


def test_handles_lines_with_just_hash():
    doc = "# @config\n# foo: bar\n#\n# baz: qux\n# @end"
    result = extract_config_block(doc)
    assert result == "foo: bar\n\nbaz: qux"


def test_returns_none_if_non_comment_before_config():
    doc = "\nfoo: bar\n# @config\n# baz: qux\n# @end"
    result = extract_config_block(doc)
    assert result is None


def test_returns_none_if_non_comment_inside_block():
    doc = "# @config\n# foo: bar\nbaz: qux\n# @end"
    result = extract_config_block(doc)
    assert result is None
