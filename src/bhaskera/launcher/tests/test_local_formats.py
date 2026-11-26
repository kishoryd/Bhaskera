"""The local dataset reader: which files it accepts and how it reads them. Ray itself is stood in for (CPU, no cluster needed);
the reading with a real Ray is checked on the cluster, where Ray is installed."""
from __future__ import annotations

import gzip
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


@pytest.fixture
def loc(monkeypatch):
    seen = {}
    ray = types.ModuleType("ray")
    data = types.ModuleType("ray.data")
    data.Dataset = object
    data.read_parquet = lambda files: ("parquet", list(files))
    data.read_json = lambda files, override_num_blocks=None: ("json", list(files))
    data.from_pandas = lambda df: ("pandas", df)
    data.read_csv = lambda *a, **k: ("csv", a)
    ray.data = data
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.data", data)
    pkg = types.ModuleType("bhaskera.data")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "bhaskera.data", pkg)
    reg = types.ModuleType("bhaskera.data.registry")
    reg.register = lambda name: (lambda f: f)
    reg.register_raw = lambda name, **k: (lambda f: f)
    tok = types.ModuleType("bhaskera.data.tokenize")
    tok.load_tokenized = tok.tokenize_dataset = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "bhaskera.data.registry", reg)
    monkeypatch.setitem(sys.modules, "bhaskera.data.tokenize", tok)
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "data" / "datasets" / "local_chat.py"
    spec = importlib.util.spec_from_file_location("local_chat_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod


def rows(n=3):
    return [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(n)]


def test_each_format_is_recognised_by_its_name(loc, tmp_path):
    for name, want in (("a.jsonl", "json"), ("a.json", "json"), ("a.ndjson", "json"), ("a.JSONL", "json"), ("a.parquet", "parquet"), ("a.csv", "csv"),
                       ("a.tsv", "tsv"), ("a.arrow", "arrow"), ("a.feather", "arrow"), ("a.csv.gz", "csv"), ("a.jsonl.gz", "json"), ("a.txt", ""), ("a.xlsx", "")):
        assert loc._kind(str(tmp_path / name)) == want, name
    d = tmp_path / "hfds"
    d.mkdir()
    assert loc._kind(str(d)) == ""
    (d / "dataset_info.json").write_text("{}")
    assert loc._kind(str(d)) == "hf"


def test_a_folder_is_searched_for_every_supported_format_and_a_saved_dataset_is_one_item(loc, tmp_path):
    for n in ("train.jsonl", "b.csv", "c.tsv", "d.parquet", "e.arrow", "f.csv.gz", "notes.txt", "x.bin"):
        (tmp_path / n).write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "g.json").write_text("[]")
    names = sorted(Path(f).name for f in loc._resolve_files(str(tmp_path)))
    assert names == ["b.csv", "c.tsv", "d.parquet", "e.arrow", "f.csv.gz", "g.json", "train.jsonl"]
    hf = tmp_path / "saved"
    hf.mkdir()
    (hf / "state.json").write_text("{}")
    assert loc._resolve_files(str(hf)) == [str(hf)]
    with pytest.raises(FileNotFoundError, match="matched no files"):
        loc._resolve_files(str(tmp_path / "nope" / "*.jsonl"))


def test_a_json_file_is_an_array_or_one_row_per_line(loc, tmp_path):
    arr = tmp_path / "arr.json"
    arr.write_text("  \n" + json.dumps(rows()))
    lines = tmp_path / "lines.jsonl"
    lines.write_text("\n".join(json.dumps(r) for r in rows()))
    gz = tmp_path / "arr.json.gz"
    gz.write_bytes(gzip.compress(json.dumps(rows()).encode()))
    assert loc._is_json_array(str(arr)) and loc._is_json_array(str(gz)) and not loc._is_json_array(str(lines))


def test_json_lines_go_to_ray_and_arrays_are_read_whole(loc, tmp_path):
    lines = tmp_path / "l.jsonl"
    lines.write_text("\n".join(json.dumps(r) for r in rows(2)))
    assert loc._read_files([str(lines)]) == ("json", [str(lines)])
    arr = tmp_path / "a.json"
    arr.write_text(json.dumps(rows(4)))
    kind, df = loc._read_files([str(arr)])
    assert kind == "pandas" and len(df) == 4 and df["messages"][0][0]["content"] == "q0"


def test_csv_and_tsv_are_read_as_text_and_list_columns_are_decoded(loc, tmp_path):
    c = tmp_path / "t.csv"
    pd.DataFrame({"instruction": ["[INST] hi", "b"], "output": ["x", "y"],
                  "messages": [json.dumps(rows(1)[0]["messages"]), "plain"]}).to_csv(c, index=False)
    kind, df = loc._read_files([str(c)])
    assert kind == "pandas" and list(df["instruction"]) == ["[INST] hi", "b"]          # an ordinary column that starts with [ is left alone
    assert df["messages"][0][1]["content"] == "a0" and df["messages"][1] == "plain"
    t = tmp_path / "t.tsv"
    pd.DataFrame({"instruction": ["a\tb", "c"], "output": ["1", "2"]}).to_csv(t, sep="\t", index=False)
    kind, df = loc._read_files([str(t)])
    assert list(df["output"]) == ["1", "2"] and df.shape == (2, 2)
    gz = tmp_path / "z.csv.gz"
    pd.DataFrame({"a": ["1"], "b": ["2"]}).to_csv(gz, index=False, compression="gzip")
    assert loc._read_files([str(gz)])[1].shape == (1, 2)


def test_empty_cells_stay_empty_text_and_not_missing_values(loc, tmp_path):
    c = tmp_path / "e.csv"
    c.write_text("instruction,input,output\nsay hi,,hi\n")
    df = loc._read_files([str(c)])[1]
    assert df["input"][0] == ""


def test_arrow_and_saved_datasets_are_read(loc, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    t = pa.table({"instruction": ["a", "b"], "output": ["c", "d"]})
    f = tmp_path / "d.feather"
    with pa.OSFile(str(f), "wb") as sink, pa.ipc.new_file(sink, t.schema) as w:
        w.write_table(t)
    assert loc._read_files([str(f)])[1].shape == (2, 2)
    g = tmp_path / "s.arrow"
    with pa.OSFile(str(g), "wb") as sink, pa.ipc.new_stream(sink, t.schema) as w:        # the stream form, as datasets writes it
        w.write_table(t)
    assert loc._read_files([str(g)])[1].shape == (2, 2)
    try:
        import datasets
    except ImportError:
        pytest.skip("datasets is not installed here")
    ds = datasets.Dataset.from_dict({"instruction": ["a", "b", "c"], "output": ["1", "2", "3"]})
    ds.save_to_disk(str(tmp_path / "saved"))
    assert loc._read_files([str(tmp_path / "saved")])[1].shape == (3, 2)
    datasets.DatasetDict({"train": ds, "validation": ds}).save_to_disk(str(tmp_path / "multi"))
    with pytest.raises(ValueError, match="several splits.*train, validation|several splits.*validation, train"):
        loc._read_files([str(tmp_path / "multi")])


def test_mixed_formats_and_unknown_files_are_refused_with_the_names(loc, tmp_path):
    a, b, c = tmp_path / "a.csv", tmp_path / "b.parquet", tmp_path / "c.txt"
    for f in (a, b, c):
        f.write_text("x")
    with pytest.raises(ValueError, match="mixed formats"):
        loc._read_files([str(a), str(b)])
    with pytest.raises(ValueError, match="unsupported file"):
        loc._read_files([str(c)])
