"""ensure_pad_token: tokenizers with no pad token, no EOS token, or neither (BharatGen Param-1-7B). CPU only."""
from __future__ import annotations

import json

import pytest

from bhaskera.data.tokenize import ensure_pad_token


class _Tok:
    def __init__(self, pad=None, eos=None):
        self.pad_token, self.eos_token = pad, eos

    def convert_ids_to_tokens(self, i):
        return f"<id{i}>"


def _model_dir(tmp_path, **cfg):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 8, **cfg}))
    return str(tmp_path)


def test_keeps_existing_pad(tmp_path):
    t = _Tok(pad="<pad>", eos="</s>")
    ensure_pad_token(t, _model_dir(tmp_path))
    assert t.pad_token == "<pad>"


def test_uses_eos_when_no_pad(tmp_path):
    t = _Tok(eos="</s>")
    ensure_pad_token(t, _model_dir(tmp_path))
    assert t.pad_token == "</s>"


def test_uses_config_eos_id_when_tokenizer_has_no_special_tokens(tmp_path):
    t = _Tok()
    ensure_pad_token(t, _model_dir(tmp_path, eos_token_id=2))
    assert t.pad_token == "<id2>"


def test_config_pad_id_zero_is_respected(tmp_path):
    t = _Tok()
    ensure_pad_token(t, _model_dir(tmp_path, pad_token_id=0, eos_token_id=2))
    assert t.pad_token == "<id0>"


def test_config_eos_list_takes_first(tmp_path):
    t = _Tok()
    ensure_pad_token(t, _model_dir(tmp_path, eos_token_id=[5, 6]))
    assert t.pad_token == "<id5>"


def test_clear_error_when_nothing_to_use(tmp_path):
    with pytest.raises(ValueError, match="no pad or EOS token"):
        ensure_pad_token(_Tok(), _model_dir(tmp_path, pad_token_id=None, eos_token_id=None))
