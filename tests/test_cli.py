from llm_compression.cli import main


def test_cli_help():
    from llm_compression.cli import parser

    assert "decompress" in parser().format_help()


def test_cli_bad_archive(tmp_path, capsys):
    source = tmp_path / "bad.llmc"
    source.write_bytes(b"not an archive")
    assert main(["info", str(source)]) == 1
    assert "Not a supported LLMC archive" in capsys.readouterr().err


def test_cli_does_not_overwrite(tmp_path, capsys):
    source = tmp_path / "input.txt"
    source.write_text("hello", encoding="utf-8")
    output = tmp_path / "output.llmc"
    output.write_bytes(b"keep me")
    assert main(["compress", str(source), str(output)]) == 1
    assert output.read_bytes() == b"keep me"
    assert "already exists" in capsys.readouterr().err


def test_cli_rejects_invalid_utf8_before_model_load(tmp_path, capsys):
    source = tmp_path / "input.txt"
    source.write_bytes(b"\xff")
    assert main(["compress", str(source), str(tmp_path / "out.llmc")]) == 1
    assert "utf-8" in capsys.readouterr().err
