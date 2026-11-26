"""Creating a recipe from a few choices: what it writes, what it refuses, and that the training loader accepts it."""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from bhaskera.launcher import launch, recipes


@pytest.fixture
def world(tmp_path):
    work = tmp_path / "work"
    (work / "recipes").mkdir(parents=True)
    (work / "submit.sh").write_text("")
    model = tmp_path / "models" / "Qwen3-14B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    cache = work / "cache"
    for name, rows in (("local_train_aaaa", 1000), ("local_val_bbbb", 50)):
        d = cache / name
        d.mkdir(parents=True)
        (d / "part-0.parquet").write_text("x")
        (d / "metadata.json").write_text(json.dumps({"model_name": str(model), "seq_len": 2048, "num_rows": rows, "dataset_name": "local"}))
    (cache / "not_a_cache").mkdir()
    return work, model, cache


def form(model_dir, cache_dir, **kw):
    return {"name": "my-run", "method": "qlora", "model": str(model_dir), "train_cache": str(cache_dir / "local_train_aaaa"),
            "val_cache": str(cache_dir / "local_val_bbbb"), **kw}


def test_caches_are_found_with_what_they_hold(world):
    work, model, cache = world
    found = recipes.find_caches(recipes.cache_folders(work, []))
    assert {(c["name"], c["side"], c["seq_len"], c["rows"], c["model"]) for c in found} == {
        ("local_train_aaaa", "train", 2048, 1000, "Qwen3-14B"), ("local_val_bbbb", "val", 2048, 50, "Qwen3-14B")}


def test_caches_named_by_existing_recipes_are_found_too(tmp_path, world):
    work, model, cache = world
    other = tmp_path / "elsewhere" / "cache"
    (other / "local_train_zzzz").mkdir(parents=True)
    (other / "local_train_zzzz" / "p.parquet").write_text("x")
    (other / "local_train_zzzz" / "metadata.json").write_text(json.dumps({"model_name": "m", "seq_len": 8, "num_rows": 1}))
    r = work / "recipes" / "old.yaml"
    r.write_text(yaml.safe_dump({"data": {"tokenized_path": str(other / "local_train_zzzz")}}))
    assert "local_train_zzzz" in {c["name"] for c in recipes.find_caches(recipes.cache_folders(work, [r]))}


def test_a_recipe_is_built_from_the_choices_and_loads_cleanly(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, steps="600", lr="2e-4", rank="32"), work)
    assert r["model"]["quantization"] == "qlora" and r["training"]["distributed"]["strategy"] == "ddp" and r["training"]["batch_size"] == 2
    assert r["lora"]["r"] == 32 and r["lora"]["alpha"] == 64 and r["training"]["max_steps"] == 600 and r["training"]["lr"] == 2e-4
    assert r["data"]["seq_len"] == 2048 and r["data"]["val_tokenized_path"].endswith("local_val_bbbb")
    assert r["checkpoint"]["save_dir"] == str((work / "checkpoints" / "my-run").resolve()), "an absolute folder, so resuming finds it from anywhere"
    assert r["checkpoint"]["enabled"] and r["checkpoint"]["save_every_steps"] <= 500 and r["logging"]["mlflow_log_adapter"] is True
    assert r["monitoring"]["dashboard_host"] == "127.0.0.1"
    recipes.validate(recipes.render(r))           # the training job's own loader: no error, nothing ignored


@pytest.mark.parametrize("method,quant,strategy", [("qlora", "qlora", "ddp"), ("ddp", "none", "ddp"), ("fsdp", "none", "fsdp")])
def test_each_method(world, method, quant, strategy):
    work, model, cache = world
    r = recipes.build(form(model, cache, method=method), work)
    assert (r["model"]["quantization"], r["training"]["distributed"]["strategy"]) == (quant, strategy)
    recipes.validate(recipes.render(r))


def test_no_validation_data_means_no_evaluation(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, val_cache=""), work)
    assert "val_tokenized_path" not in r["data"] and r["training"]["eval_steps"] == 0
    recipes.validate(recipes.render(r))


@pytest.mark.parametrize("change,word", [
    ({"name": "bad name"}, "recipe name"), ({"name": ""}, "recipe name"), ({"method": "magic"}, "choose a method"),
    ({"model": "relative/path"}, "full path"), ({"model": "/does/not/exist"}, "not a folder"), ({"model": ""}, "missing"),
    ({"seq_len": "4096"}, "tokenised with sequence length 2048"), ({"steps": "0"}, "between"), ({"lr": "5"}, "between"),
    ({"lr": "fast"}, "must be a number"), ({"rank": "9999"}, "between"), ({"train_cache": ""}, "missing"),
])
def test_bad_choices_are_explained(world, change, word):
    work, model, cache = world
    with pytest.raises(recipes.RecipeError, match=word):
        recipes.build(form(model, cache, **change), work)


def test_a_model_folder_without_config_and_a_folder_that_is_not_a_cache_are_refused(world, tmp_path):
    work, model, cache = world
    empty = tmp_path / "empty-model"
    empty.mkdir()
    with pytest.raises(recipes.RecipeError, match="config.json"):
        recipes.build(form(empty, cache), work)
    with pytest.raises(recipes.RecipeError, match="not a tokenised cache"):
        recipes.build(form(model, cache, train_cache=str(cache / "not_a_cache")), work)


def test_a_recipe_is_never_overwritten(world):
    work, model, cache = world
    text = recipes.render(recipes.build(form(model, cache), work))
    p = recipes.write(work / "recipes", "my-run", text)
    assert yaml.safe_load(p.read_text())["logging"]["run_name"] == "my-run"
    with pytest.raises(recipes.RecipeError, match="exists already"):
        recipes.write(work / "recipes", "my-run", "other: 1\n")
    assert "other" not in p.read_text()


def test_validation_catches_an_ignored_option():
    with pytest.raises(recipes.RecipeError, match="do nothing"):
        recipes.validate("model:\n  name: x\n  quantisation: qlora\n")


def test_raw_data_becomes_a_tokenising_recipe(world, tmp_path):
    work, model, cache = world
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n")
    name, text = recipes.tokenize_recipe({"name": "my-data", "model": str(model), "train_path": str(train), "format": "alpaca", "seq_len": "2048"}, work)
    d = yaml.safe_load(text)
    assert name == "tokenize-my-data.yaml" and d["data"]["format"] == "alpaca" and d["data"]["train_path"] == str(train)
    assert d["data"]["cache_dir"] == str((work / "cache").resolve()) and d["data"]["seq_len"] == 2048
    for bad, word in (({"train_path": ""}, "missing"), ({"train_path": "rel.jsonl"}, "full path"), ({"format": "xml"}, "chatml"), ({"name": "x y"}, "name")):
        with pytest.raises(recipes.RecipeError, match=word):
            recipes.tokenize_recipe({"name": "n", "model": str(model), "train_path": str(train), **bad}, work)


class Fake:
    def __init__(self, out="submitted job 31: tokenise"):
        self.calls, self.out = [], out

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, self.out, "")


def launcher(work, fake):
    return launch.Launcher(work, runs=lambda: [], env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a"}, run=fake)


def test_the_page_offers_caches_models_and_methods(world):
    work, model, cache = world
    (work / "recipes" / "ready.yaml").write_text(yaml.safe_dump({"model": {"name": str(model)}}))
    o = launcher(work, Fake()).options()
    assert {c["name"] for c in o["caches"]} == {"local_train_aaaa", "local_val_bbbb"} and o["model_folders"] == [str(model)]
    assert set(o["methods"]) == {"qlora", "ddp", "fsdp", "hsdp", "pipeline", "custom"} and o["methods"]["qlora"]["label"] == "QLoRA + DDP"


def test_creating_a_recipe_from_the_page(world):
    work, model, cache = world
    fake = Fake()
    m = launcher(work, fake)
    preview = m.launch({"action": "check", "kind": "recipe", **form(model, cache)})
    assert preview["ok"] and preview["dry"] and "max_steps" in preview["output"] and not (work / "recipes" / "my-run.yaml").exists()
    out = m.launch({"action": "start", "kind": "recipe", **form(model, cache)})
    assert out["ok"] and out["recipe"] == "my-run.yaml" and (work / "recipes" / "my-run.yaml").is_file() and fake.calls == [], "no job: only a file"
    with pytest.raises(launch.LaunchError, match="exists already"):
        m.launch({"action": "start", "kind": "recipe", **form(model, cache)})
    with pytest.raises(launch.LaunchError, match="full path"):
        m.launch({"action": "start", "kind": "recipe", **form(model, cache, model="rel")})
    assert any(r["file"] == "my-run.yaml" and r["ready"] for r in m.options()["recipes"]), "the new recipe is ready to start at once"


def test_preparing_data_from_the_page(world, tmp_path):
    work, model, cache = world
    train = tmp_path / "t.jsonl"
    train.write_text("{}\n")
    fake = Fake()
    out = launcher(work, fake).launch({"action": "start", "kind": "tokenize", "name": "d1", "model": str(model), "train_path": str(train), "format": "chatml"})
    argv = fake.calls[-1][0]
    assert argv[0] == "bash" and argv[2:] == ["tokenize", str(work / "recipes" / "tokenize-d1.yaml")] and out["job"] == "31"
    assert (work / "recipes" / "tokenize-d1.yaml").is_file()


def test_hsdp_and_pipeline_set_their_own_keys(world):
    work, model, cache = world
    h = recipes.build(form(model, cache, method="hsdp", hsdp_shard_size="4"), work)
    assert h["training"]["distributed"] == {"strategy": "fsdp", "fsdp": {"sharding_strategy": "HYBRID_SHARD", "hsdp_shard_size": 4, "activation_checkpointing": True}}
    p = recipes.build(form(model, cache, method="pipeline", stages="4"), work)
    assert p["training"]["distributed"]["pipeline"]["stages"] == 4 and p["training"]["distributed"]["pipeline"]["microbatches"] == 8
    recipes.validate(recipes.render(h))


def test_multimodal_recipe_has_media_and_refuses_what_cannot_work(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, method="fsdp", model_kind="multimodal", modalities=["image", "audio"]), work)
    assert r["model"]["kind"] == "multimodal" and r["data"]["modalities"] == ["text", "image", "audio"] and "runtime_packing" not in r["data"]
    for method in ("qlora", "pipeline"):
        with pytest.raises(recipes.RecipeError, match="not available for multimodal"):
            recipes.build(form(model, cache, method=method, model_kind="multimodal"), work)
    with pytest.raises(recipes.RecipeError, match="image, video and audio"):
        recipes.build(form(model, cache, method="fsdp", model_kind="multimodal", modalities=["smell"]), work)


def test_every_setting_can_be_set_and_typos_are_refused(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, method="fsdp", settings={"training.weight_decay": "0.01", "training.distributed.fsdp.cpu_offload": "yes",
                                                                    "logging.tags": "a, b", "monitoring.dashboard_host": "0.0.0.0"}), work)
    assert r["training"]["weight_decay"] == 0.01 and r["training"]["distributed"]["fsdp"]["cpu_offload"] is True and r["logging"]["tags"] == ["a", "b"]
    assert r["monitoring"]["dashboard_host"] == "127.0.0.1"
    with pytest.raises(recipes.RecipeError, match="did you mean 'training.lr'"):
        recipes.build(form(model, cache, method="fsdp", settings={"training.lrr": "1"}), work)
    with pytest.raises(recipes.RecipeError, match="whole number"):
        recipes.build(form(model, cache, method="fsdp", settings={"training.max_steps": "x"}), work)
    assert len(recipes.fields()) > 100 and all(g["fields"] for g in recipes.schema())


def test_custom_recipe_from_yaml(world):
    work, model, cache = world
    text = yaml.safe_dump({"model": {"name": str(model)}, "checkpoint": {"save_dir": "/elsewhere"}, "logging": {"run_name": "old"}})
    r = recipes.build({"name": "mine", "method": "custom", "yaml": text, "copy": True}, work)
    assert r["logging"]["run_name"] == "mine" and r["checkpoint"]["save_dir"].endswith("checkpoints/mine")
    with pytest.raises(recipes.RecipeError, match="placeholder"):
        recipes.build({"name": "mine", "method": "custom", "yaml": "model:\n  name: /scratch/<project>/m\n"}, work)
    with pytest.raises(recipes.RecipeError, match="not valid YAML"):
        recipes.build({"name": "mine", "method": "custom", "yaml": "a: [1"}, work)


def test_ddp_recipes_checkpoint_activations(world):
    work, model, cache = world
    for method in ("qlora", "ddp"):
        r = recipes.build(form(model, cache, method=method), work)
        assert r["training"]["distributed"]["ddp"] == {"activation_checkpointing": True}
    recipes.validate(recipes.render(recipes.build(form(model, cache, method="qlora"), work)))


def _model_dir(base, name, config, size=1000):
    d = base / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    (d / "model-00001.safetensors").write_bytes(b"x" * size)
    return d


def test_model_catalog_lists_shared_models_with_their_kind_and_size(tmp_path):
    shared = tmp_path / "shared"
    _model_dir(shared, "Qwen3-14B", {"architectures": ["Qwen3ForCausalLM"]}, 2_000_000)
    _model_dir(shared, "Qwen3-VL-2B", {"architectures": ["Qwen3VLForConditionalGeneration"], "vision_config": {}})
    _model_dir(shared, "Omni", {"architectures": ["Qwen2_5OmniForConditionalGeneration"], "thinker_config": {}})
    (shared / "not-a-model").mkdir()
    mine = _model_dir(tmp_path / "mine", "my-own", {"architectures": ["LlamaForCausalLM"]})
    cat = recipes.model_catalog([shared], [str(mine), str(shared / "Qwen3-14B")])          # the last one is also shared: listed once
    assert [(m["name"], m["kind"], m["group"]) for m in cat] == [("Omni", "multimodal", "shared"), ("Qwen3-14B", "text", "shared"), ("Qwen3-VL-2B", "multimodal", "shared"), ("my-own", "text", "yours")]
    assert next(m for m in cat if m["name"] == "Qwen3-14B")["size_gb"] == 0.0 and all(m["arch"] for m in cat)


def test_shared_folders_come_from_the_environment_or_sit_next_to_the_install(tmp_path):
    (tmp_path / "apps" / "models").mkdir(parents=True)
    (tmp_path / "apps" / "data").mkdir()
    (tmp_path / "apps" / "1.0.0").mkdir()
    env = {"BHASKERA_HOME": str(tmp_path / "apps" / "1.0.0")}
    assert recipes.shared_folders(env, "models") == [tmp_path / "apps" / "models"] and recipes.shared_folders(env, "data") == [tmp_path / "apps" / "data"]
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert recipes.shared_folders({**env, "BHASKERA_MODELS": f"{other}:/does/not/exist"}, "models") == [other]
    assert recipes.shared_folders({}, "models") == []


def test_the_page_offers_shared_models_and_marks_shared_data(world, tmp_path):
    work, model, cache = world
    shared = tmp_path / "apps" / "models"
    _model_dir(shared, "Shared-Model", {"architectures": ["LlamaForCausalLM"]})
    data = tmp_path / "apps" / "data"
    d = data / "local_train_shared"
    d.mkdir(parents=True)
    (d / "part-0.parquet").write_text("x")
    (d / "metadata.json").write_text(json.dumps({"model_name": "/m/Shared-Model", "seq_len": 1024, "num_rows": 5}))
    (work / "recipes" / "ready.yaml").write_text(yaml.safe_dump({"model": {"name": str(model)}}))
    L = launch.Launcher(work, runs=lambda: [], env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a", "BHASKERA_HOME": str(tmp_path / "apps" / "1.0.0")})
    o = L.options()
    assert [(m["name"], m["group"]) for m in o["model_catalog"]] == [("Shared-Model", "shared"), ("Qwen3-14B", "yours")]
    flags = {c["name"]: c["shared"] for c in o["caches"]}
    assert flags["local_train_shared"] is True and flags["local_train_aaaa"] is False


def test_your_copy_of_a_shared_model_or_cache_is_not_offered_twice(world, tmp_path):
    work, model, cache = world
    shared = tmp_path / "apps" / "models"
    _model_dir(shared, "Qwen3-14B", {"architectures": ["Qwen3ForCausalLM"]})
    _model_dir(tmp_path / "other", "Qwen3-14B", {"architectures": ["Qwen3ForCausalLM"]})          # same name and size as the shared one
    bigger = _model_dir(tmp_path / "big", "Qwen3-14B", {"architectures": ["Qwen3ForCausalLM"]})
    with open(bigger / "model-00001.safetensors", "r+b") as f:
        f.truncate(300_000_000)                                                                          # same name, a different model: kept
    cat = recipes.model_catalog([shared], [str(tmp_path / "other" / "Qwen3-14B"), str(bigger)])
    assert [(m["name"], m["group"], m["size_gb"]) for m in cat] == [("Qwen3-14B", "shared", 0.0), ("Qwen3-14B", "yours", 0.3)]
    data = tmp_path / "apps" / "data"
    for name in ("local_train_aaaa",):
        d = data / name
        d.mkdir(parents=True)
        (d / "p.parquet").write_text("x")
        (d / "metadata.json").write_text(json.dumps({"model_name": str(model), "seq_len": 2048, "num_rows": 1000}))
    L = launch.Launcher(work, runs=lambda: [], env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a", "BHASKERA_HOME": str(tmp_path / "apps" / "1.0.0")})
    names = [(c["name"], c["shared"]) for c in L.options()["caches"]]
    assert names.count(("local_train_aaaa", True)) == 1 and ("local_train_aaaa", False) not in names and ("local_val_bbbb", False) in names


def test_a_model_that_ships_its_own_code_gets_trust_remote_code(world):
    work, model, cache = world
    own = _model_dir(model.parent, "Param-like", {"architectures": ["ParamForCausalLM"], "auto_map": {"AutoModelForCausalLM": "modeling_param.ParamForCausalLM"}})
    assert recipes.needs_remote_code(str(own)) is True and recipes.needs_remote_code(str(model)) is False and recipes.needs_remote_code("/does/not/exist") is False
    assert recipes.build(form(own, cache, method="fsdp"), work)["model"]["trust_remote_code"] is True
    assert recipes.build(form(model, cache, method="fsdp"), work)["model"]["trust_remote_code"] is False
    assert next(m for m in recipes.model_catalog([own.parent], []) if m["name"] == "Param-like")["custom_code"] is True
    assert yaml.safe_load(recipes.tokenize_recipe({"name": "t", "model": str(own), "train_path": str(work / "submit.sh")}, work)[1].split("\n", 1)[1])["model"]["trust_remote_code"] is True


def test_a_cache_is_train_or_validation_by_the_words_in_its_name_or_its_metadata():
    side = recipes.cache_side
    for name, want in (("local_train_ab12", "train"), ("local_val_cd34", "val"), ("nemotron-chat_qwen3-14b_seq4096_train", "train"),
                       ("nemotron-chat_qwen3-14b_seq4096_val", "val"), ("train", "train"), ("my.validation.set", "val"), ("EVAL_x", "val"),
                       ("pokemon_train_x32", "train"), ("trainer-notes", ""), ("interval", ""), ("data", "")):
        assert side(name) == want, name


def test_the_side_falls_back_to_the_dataset_name_in_the_metadata(tmp_path):
    d = tmp_path / "chat-qwen-4096"
    d.mkdir()
    (d / "p.parquet").write_text("x")
    (d / "metadata.json").write_text(json.dumps({"model_name": "m", "seq_len": 8, "num_rows": 1, "dataset_name": "local_val"}))
    assert recipes.cache_info(d)["side"] == "val"


def test_checkpoints_can_be_switched_off_and_their_number_and_interval_chosen(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, save_every_steps="200", keep_last_n="5"), work)
    assert r["checkpoint"]["enabled"] is True and r["checkpoint"]["save_every_steps"] == 200 and r["checkpoint"]["keep_last_n"] == 5
    recipes.validate(recipes.render(r))
    r = recipes.build(form(model, cache, checkpoints=False, name="no-ckpt"), work)
    assert r["checkpoint"]["enabled"] is False and r["checkpoint"]["keep_last_n"] == 3          # the number is kept for when it is turned on
    recipes.validate(recipes.render(r))
    assert recipes.build(form(model, cache, keep_last_n="0", name="keep-all"), work)["checkpoint"]["keep_last_n"] == 0
    for bad in ({"checkpoints": "no"}, {"keep_last_n": "-1"}, {"keep_last_n": "x"}, {"keep_last_n": "5000"}):
        with pytest.raises(recipes.RecipeError):
            recipes.build(form(model, cache, name="bad", **bad), work)


def test_a_full_fine_tune_trains_every_weight_and_needs_its_checkpoints(world):
    work, model, cache = world
    r = recipes.build(form(model, cache, method="fsdp", tuning="full", name="full-ft"), work)
    assert r["lora"] == {"enabled": False} and r["training"]["distributed"]["strategy"] == "fsdp" and r["model"]["quantization"] == "none"
    assert r["training"]["lr"] == 1e-5 and r["logging"]["mlflow_log_adapter"] is False and r["checkpoint"]["enabled"] is True      # a lower rate; no adapter to keep
    recipes.validate(recipes.render(r))
    assert recipes.build(form(model, cache, method="fsdp", tuning="full", lr="2e-5", name="ft2"), work)["training"]["lr"] == 2e-5
    ddp = recipes.build(form(model, cache, method="ddp", tuning="full", name="ft-ddp"), work)
    assert ddp["lora"]["enabled"] is False
    hs = recipes.build(form(model, cache, method="hsdp", tuning="full", name="ft-hs"), work)
    assert hs["training"]["distributed"]["fsdp"]["sharding_strategy"] == "HYBRID_SHARD" and hs["lora"]["enabled"] is False
    assert recipes.build(form(model, cache, method="fsdp", name="lora-fsdp"), work)["lora"]["enabled"] is True      # LoRA stays the default


def test_full_fine_tuning_is_refused_where_it_cannot_work_and_cannot_have_checkpoints_off(world):
    work, model, cache = world
    with pytest.raises(recipes.RecipeError, match="QLoRA.*cannot train every weight"):
        recipes.build(form(model, cache, tuning="full", name="a"), work)                       # the form's default method is QLoRA
    with pytest.raises(recipes.RecipeError, match="Pipeline trains a LoRA add-on only"):
        recipes.build(form(model, cache, method="pipeline", tuning="full", name="b"), work)
    with pytest.raises(recipes.RecipeError, match="only in its checkpoints"):
        recipes.build(form(model, cache, method="fsdp", tuning="full", checkpoints=False, name="c"), work)
    with pytest.raises(recipes.RecipeError, match="choose what to train"):
        recipes.build(form(model, cache, tuning="everything", name="d"), work)
