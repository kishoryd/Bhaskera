"""Saving a trained model in the MLflow registry and finding it again, with a stand-in client (no MLflow needed)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from bhaskera.launcher import evaluate, register

RUN = "a" * 32
RUN2 = "b" * 32


class FakeClient:
    def __init__(self, with_adapter=True):
        self.models, self.versions, self.aliases = {}, [], {}
        self.with_adapter = with_adapter
        self.runs = {r: NS(info=NS(run_name=f"run-{r[:2]}", artifact_uri=f"/store/1/{r}/artifacts"),
                          data=NS(metrics={"eval_loss": 1.02, "eval_accuracy": 0.7176, "lmeval_arc_easy_acc_norm_tuned": 0.83,
                                           "lmeval_arc_easy_acc_norm_delta": 0.015, "loss": 0.9}, params={"model.name": "/m/Qwen3-14B", "lora.r": "16", "lora.alpha": "32", "model.quantization": "qlora", "training.max_steps": "500", "training.lr": "0.0001"}))
                     for r in (RUN, RUN2)}

    def get_run(self, rid):
        if rid not in self.runs:
            raise KeyError(rid)
        return self.runs[rid]

    def list_artifacts(self, rid, path):
        return [NS(path="adapter/adapter_config.json")] if self.with_adapter else []

    def create_registered_model(self, name, description=""):
        if name in self.models:
            raise RuntimeError("exists")
        self.models[name] = description

    def create_model_version(self, name, source, run_id, tags, description=""):
        v = NS(name=name, version=str(1 + sum(1 for x in self.versions if x.name == name)), source=source, run_id=run_id, tags=dict(tags), aliases=[], description=description)
        self.versions.append(v)
        return v

    def set_registered_model_alias(self, name, alias, version):
        for v in self.versions:
            if v.name == name:
                v.aliases = [a for a in v.aliases if a != alias]
                if v.version == str(version):
                    v.aliases.append(alias)

    def search_registered_models(self):
        return [NS(name=n) for n in self.models]

    def search_model_versions(self, flt):
        name = flt.split("'")[1]
        return [v for v in self.versions if v.name == name]

    def get_model_version_by_alias(self, name, alias):
        for v in self.versions:
            if v.name == name and alias in v.aliases:
                return v
        raise KeyError(alias)

    def get_model_version(self, name, version):
        for v in self.versions:
            if v.name == name and v.version == str(version):
                return v
        raise KeyError(version)


def test_add_registers_a_version_that_says_what_it_is():
    c = FakeClient()
    r = register.add(c, RUN, "my-model", alias="champion")
    assert r == {"name": "my-model", "version": "1", "alias": "champion", "run_id": RUN}
    v = c.versions[0]
    assert v.source == f"/store/1/{RUN}/artifacts/adapter" and v.run_id == RUN
    assert v.tags["base_model"] == "/m/Qwen3-14B" and v.tags["eval_accuracy"] == "0.7176" and v.tags["lmeval_arc_easy_acc_norm_tuned"] == "0.8300"
    assert "loss" not in v.tags, "only the evaluation and benchmark numbers are kept"


def test_a_second_version_and_a_moved_alias():
    c = FakeClient()
    register.add(c, RUN, "my-model", alias="champion")
    register.add(c, RUN2, "my-model")
    assert [v.version for v in c.versions] == ["1", "2"]
    register.add(c, RUN2, "my-model", alias="champion")                 # version 3 is now the champion
    assert [v.aliases for v in c.versions] == [[], [], ["champion"]]


@pytest.mark.parametrize("run,name,alias,word", [
    ("short", "m", None, "32-character"), (RUN, "bad name", None, "model name"), (RUN, "m", "1bad", "alias"), ("c" * 32, "m", None, "no run"),
])
def test_bad_input_is_explained(run, name, alias, word):
    with pytest.raises(register.RegisterError, match=word):
        register.add(FakeClient(), run, name, alias)


def test_a_run_without_an_adapter_is_refused():
    with pytest.raises(register.RegisterError, match="mlflow_log_adapter"):
        register.add(FakeClient(with_adapter=False), RUN, "m")


def test_listing_and_resolving_names():
    c = FakeClient()
    register.add(c, RUN, "my-model", alias="champion")
    register.add(c, RUN2, "my-model")
    models = register.listing(c)
    assert models[0]["name"] == "my-model" and [v["version"] for v in models[0]["versions"]] == ["2", "1"]
    assert models[0]["versions"][1]["aliases"] == ["champion"]
    assert register.resolve(c, "my-model@champion") == RUN
    assert register.resolve(c, "my-model:2") == RUN2
    assert register.resolve(c, "my-model") == RUN2, "no version given: the newest"
    assert register.resolve(c, RUN) == RUN, "a run id stays a run id"
    for bad in ("nope", "my-model@missing", "my-model:9"):
        with pytest.raises(register.RegisterError):
            register.resolve(c, bad)
    assert "version   2" in register._show(models) and "<- champion" in register._show(models)
    assert "no models" in register._show([])


def test_evaluate_and_serve_find_the_adapter_by_registry_name(tmp_path, monkeypatch):
    run = tmp_path / "1" / RUN
    (run / "params").mkdir(parents=True)
    (run / "artifacts" / "adapter").mkdir(parents=True)
    (run / "meta.yaml").write_text("run_name: x\n")
    (run / "params" / "model.name").write_text("/models/Qwen3-14B")
    (run / "artifacts" / "adapter" / "adapter_config.json").write_text("{}")
    c = FakeClient()
    register.add(c, RUN, "my-model", alias="champion")
    monkeypatch.setattr(register, "make_client", lambda store: c)
    for ref in (RUN, "my-model@champion", "my-model:1", "my-model"):
        model, adapter = evaluate.from_run(tmp_path, ref)
        assert model == "/models/Qwen3-14B" and adapter == str(run / "artifacts" / "adapter")
    with pytest.raises(evaluate.EvalError, match="no run nothing"):
        evaluate.from_run(tmp_path, "nothing")


def test_a_registered_version_carries_a_hand_off_note():
    c = FakeClient()
    register.add(c, RUN, "my-model", note="For support questions. Weak on legal text.")
    text = c.versions[0].description
    assert text.startswith("For support questions. Weak on legal text.")
    for part in ("Base model: /m/Qwen3-14B", f"({RUN})", "LoRA rank 16, alpha 32", "QLoRA (4-bit base)", "500 steps", "eval loss 1.0200", "arc easy acc norm 0.830", "same path"):
        assert part in text, part
    assert "Results:" in register.model_card(NS(info=NS(run_name=""), data=NS(params={}, metrics={})), RUN) and "not evaluated yet" in register.model_card(NS(info=NS(run_name=""), data=NS(params={}, metrics={})), RUN)
    assert register.listing(c)[0]["versions"][0]["description"] == text


def test_a_benchmark_by_registry_name_is_stored_with_the_run_id(tmp_path, monkeypatch):
    from types import SimpleNamespace
    run = tmp_path / "1" / RUN
    (run / "params").mkdir(parents=True)
    (run / "artifacts" / "adapter").mkdir(parents=True)
    (run / "meta.yaml").write_text("run_name: x\n")
    (run / "params" / "model.name").write_text("/models/Qwen3-14B")
    (run / "artifacts" / "adapter" / "adapter_config.json").write_text("{}")
    c = FakeClient()
    register.add(c, RUN, "my-model", alias="champion")
    monkeypatch.setattr(register, "make_client", lambda store: c)
    stored = {}
    monkeypatch.setattr(evaluate, "tasks_dir", lambda given=None: None)
    monkeypatch.setattr(evaluate, "check_tasks", lambda *a, **k: None)
    monkeypatch.setattr(evaluate, "load_harness", lambda: (None, None))
    monkeypatch.setattr(evaluate, "evaluate_pair", lambda *a, **k: {"comparison": []})
    monkeypatch.setattr(evaluate, "table", lambda rows: "")
    monkeypatch.setattr(evaluate, "log_to_mlflow", lambda result, store, run_id, **k: stored.setdefault("run", run_id) or "x")
    args = SimpleNamespace(tasks="a", model=None, adapter=None, run="my-model@champion", store=str(tmp_path), skip_base=False, tasks_dir=None, save_samples=False,
                           samples_dir=None, out=str(tmp_path / "out.json"), limit=None, num_fewshot=None, batch_size="auto", dtype="bfloat16", four_bit=False,
                           device=None, mlflow=False)
    assert evaluate.cmd_run(args) == 0
    assert stored["run"] == RUN
