def command_file_header(doc: str) -> str:
    """
    Remove all lines, following and including the first single line
    equal to `outcome:` (with possibly trailing whitespace).
    """
    lines = doc.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "outcome:":
            return "\n".join(lines[:idx])
    return doc
