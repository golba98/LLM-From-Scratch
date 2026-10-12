"""Offline preparation fixtures never access real checkpoints or CUDA."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from llm_data.token_data import MemmapTokenDataset, create_token_dataloader
from llm_data.token_views import IndexedTokenWindowDataset
from llm_training.config import load_config
from llm_training.training import CyclingDataIterator, cosine_learning_rate

from codexa_workspace import pretraining_readiness as preparation
from codexa_workspace.chat_evaluation import fingerprint


def write_index(tmp_path, rows):
    path = tmp_path/"index.json"
    path.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def documents():
    return [dict(document_id=str(i), document_ordinal=i, source=source,
                 token_start=i*12, token_end=(i+1)*12, total_stored_token_count=12)
            for i, source in enumerate(["fineweb_edu", "wikipedia"]*4)]


def test_index_continuity_counts_and_hash(tmp_path):
    path, digest = write_index(tmp_path, documents())
    observations = {}
    assert len(list(preparation.index_rows(path, token_count=96,
                                          expected_sha256=digest, observations=observations))) == 8
    assert observations["tokens_by_source"] == {"fineweb_edu":48, "wikipedia":48}
    with pytest.raises(ValueError, match="checksum"):
        list(preparation.index_rows(path, token_count=96, expected_sha256="wrong", observations={}))


@pytest.mark.parametrize("field,value", [("token_start", 13), ("token_end", 1000),
                                         ("document_ordinal", 8), ("source", "unknown"),
                                         ("document_id", None)])
def test_malformed_index_rejected(tmp_path, field, value):
    rows = documents()
    rows[1][field] = value
    path, digest = write_index(tmp_path, rows)
    with pytest.raises(ValueError):
        list(preparation.index_rows(path, token_count=96, expected_sha256=digest, observations={}))


def test_historical_fingerprint_and_version():
    generator = torch.Generator().manual_seed(42)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected = torch.randperm(20, generator=generator)[:7].numpy()
    record = dict(epoch=0, seed=42, total_windows=20, cursor=7, pytorch=torch.__version__,
                  prefix_sha256=hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest())
    assert preparation.reconstruct_exposure(record).tolist() == sorted(expected.tolist())
    with pytest.raises(ValueError, match="fingerprint"):
        preparation.reconstruct_exposure(dict(record, prefix_sha256="wrong"))
    with pytest.raises(ValueError, match="version"):
        preparation.reconstruct_exposure(dict(record, pytorch="different"))


def test_seeded_candidates_exclude_all_document_context():
    args = dict(context=4, window_count=24, exposed=[0], pool_per_source=2, seed=42)
    first = preparation.select_candidates(iter(documents()), **args)
    second = preparation.select_candidates(iter(documents()), **args)
    assert first == second and len(first) == 4
    assert all(row["document_id"] != "0" for row in first)
    assert preparation.interval_exposed(4, 5, 4, 24, [0])
    assert not preparation.interval_exposed(5, 8, 4, 24, [0])


def test_duplicate_family_blocks_exposed_variants():
    rows = [dict(normalized_sha256="same", sketch=[1, 2], words=100),
            dict(normalized_sha256="same", sketch=[1, 2], words=100),
            dict(normalized_sha256="other", sketch=[3, 4], words=100)]
    groups, edges = preparation.duplicate_groups(rows, [{1, 2}, {1, 2}, {3, 4}])
    assert groups[0] == groups[1] != groups[2]
    assert edges[0]["kind"] == "normalized_exact"
    rows = [dict(source="fineweb_edu", selection_rank=1, historical_overlap=True),
            dict(source="fineweb_edu", selection_rank=2, historical_overlap=False),
            dict(source="wikipedia", selection_rank=3, historical_overlap=False)]
    with pytest.raises(ValueError, match="Insufficient"):
        preparation.select_validation(rows, groups, per_source=1)


def test_near_duplicate_requires_high_content_overlap():
    first = set(range(100))
    near = set(range(95)) | set(range(100, 105))
    separate = set(range(32)) | set(range(200, 300))
    rows = [dict(normalized_sha256=str(i), words=100, sketch=sorted(s)[:64])
            for i, s in enumerate([first, near, separate])]
    groups, edges = preparation.duplicate_groups(rows, [first, near, separate])
    assert groups[0] == groups[1] != groups[2]
    assert any(edge["kind"] == "near_duplicate" for edge in edges)


def test_training_excludes_target_boundary_and_historical_positions():
    selected = preparation.training_selection(8, 4, np.array([6], dtype="uint32"), [(4, 5)])
    assert selected.tolist() == [2, 3, 4, 5, 7]


def test_source_target_accounting_matches_exhaustive_shifted_positions():
    keep = np.array([True, False, True, True, False])
    prefix = np.concatenate([np.zeros(1, dtype="uint32"), np.cumsum(keep, dtype="uint32")])
    targets = {i*4+j for i, selected in enumerate(keep) if selected for j in range(1, 5)}
    for position in range(25):
        assert preparation.targets_before(position, keep, prefix, 4) == sum(t < position for t in targets)


def test_indexed_iterator_restores_cursor(tmp_path):
    file = tmp_path/"tokens.bin"
    np.arange(33, dtype="uint16").tofile(file)
    base = MemmapTokenDataset(file, dtype="uint16", context_length=4)
    view = IndexedTokenWindowDataset(base, np.array([0, 2, 3, 5, 7]), vocab_size=64)
    def iterator(cursor=0):
        return CyclingDataIterator(create_token_dataloader(view, batch_size=1, shuffle=True,
            seed=42), seed=42, batches_in_epoch=cursor)
    continuous = iterator()
    continuous.next(); continuous.next()
    restored = iterator(2)
    for _ in range(3):
        x, y = continuous.next()
        a, b = restored.next()
        assert torch.equal(x, a) and torch.equal(y, b)


def test_proposed_configuration_and_schedule(tmp_path):
    architecture = dict(vocab_size=16, context_length=4, num_layers=1, hidden_size=8,
                        num_heads=2, intermediate_size=16, dropout=0., tie_embeddings=True,
                        position_embedding_type="learned", rope_theta=10000.)
    config_path = tmp_path/"config.yaml"
    config, spec = preparation.pilot_spec(architecture, "selection", config_path,
        dict(path="base.pt", sha256=preparation.BASE_SHA256), Path("tokenizer.json"))
    config_path.write_text(yaml.safe_dump(config))
    loaded = load_config(config_path)
    assert loaded.training.max_steps == 256 and loaded.training.gradient_accumulation_steps == 32
    preparation.validate_proposal(spec)
    values = [cosine_learning_rate(step, warmup_steps=20, max_steps=256,
        peak_learning_rate=2e-5, minimum_learning_rate=2e-6) for step in range(257)]
    assert values[20] == pytest.approx(2e-5) and values[-1] == pytest.approx(2e-6)
    assert all(values[i] >= values[i+1] for i in range(20, 256))
    for changed in [dict(spec, execution_enabled=True), dict(spec, checkpoint_backend="optimized"),
                    dict(spec, maximum_targets=1)]:
        with pytest.raises(ValueError):
            preparation.validate_proposal(changed)


def test_budget_arithmetic_is_not_fresh_throughput():
    estimates = preparation.budgets()
    assert estimates["targets_per_update"] == 65536
    assert estimates["pilot_targets"] == 16777216
    assert estimates["ten_hour_scenarios"][0]["targets"] == 288108000
    assert estimates["ten_hour_scenarios"][1]["targets"] == 259297200


def test_objective_answer_check_rejects_keywords_in_contradictions():
    factual = dict(category="factual", expected="Jupiter", prompt="Largest planet?")
    assert preparation.conservative_answer_check(factual, "The answer is Jupiter.")["reference_consistency"] is True
    assert preparation.conservative_answer_check(factual, "Jupiter is mentioned, but Earth is largest.")["reference_consistency"] is None
    arithmetic = dict(category="arithmetic", expected="62", prompt="What is 53 plus 9? Output only the number.")
    assert preparation.conservative_answer_check(arithmetic, "The result is 62.")["reference_consistency"] is True
    assert preparation.conservative_answer_check(arithmetic, "The result is 42.")["reference_consistency"] is False
    assert preparation.conservative_answer_check(arithmetic, "62, but maybe 42.")["reference_consistency"] is None
    assert preparation.conservative_answer_check(dict(arithmetic, expected="99"), "99")["reference_verified"] is False


def test_selection_integrity_and_leakage(tmp_path):
    arrays = {"train_window_ids.npy":np.array([2, 3, 5], dtype="uint32"),
              "validation_window_ids.npy":np.array([1], dtype="uint32"),
              "historical_window_ids.npy":np.array([6], dtype="uint32")}
    for name, values in arrays.items():
        np.save(tmp_path/name, values)
    record = dict(format="codexa-indexed-pretraining-selection-v1", execution_enabled=False,
        eligible_training_windows=3, held_out_windows=1, historical_windows=1, excluded_intervals=[(4, 5)],
        context_length=4, index_observations=dict(windows=8),
        view_arrays={name:dict(path=str(tmp_path/name), sha256=preparation.file_sha256(tmp_path/name))
                     for name in arrays})
    record["selection_sha256"] = fingerprint(record)
    (tmp_path/"selection-manifest.json").write_text(json.dumps(record))
    preparation.validate_selection(tmp_path)
    np.save(tmp_path/"train_window_ids.npy", np.array([1, 2, 3], dtype="uint32"))
    with pytest.raises(ValueError, match="array identity"):
        preparation.validate_selection(tmp_path)
    record["view_arrays"]["train_window_ids.npy"]["sha256"] = preparation.file_sha256(tmp_path/"train_window_ids.npy")
    record["selection_sha256"] = fingerprint({k:v for k,v in record.items() if k != "selection_sha256"})
    (tmp_path/"selection-manifest.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="leakage"):
        preparation.validate_selection(tmp_path)


@pytest.mark.usefixtures("development_training_environment")
def test_tiny_cpu_optimizer_continuation_uses_indexed_cursor(tmp_path):
    from llm_architecture.model import LanguageModel
    from llm_training.checkpointing import build_checkpoint_payload, load_checkpoint, SchedulerState
    from llm_training.recovery import RecoveryStore
    from llm_training.training import train_model, resolve_precision, set_deterministic_seed

    config_value = dict(model=dict(vocab_size=64, context_length=4, num_layers=1,
        hidden_size=8, num_heads=2, intermediate_size=16, dropout=0., tie_embeddings=True,
        position_embedding_type="learned", rope_theta=10000.),
        training=dict(micro_batch_size=1, gradient_accumulation_steps=2, learning_rate=.001,
        weight_decay=.1, warmup_steps=0, max_steps=3, gradient_clip=1., precision="fp32",
        checkpoint_interval=1, evaluation_interval=1, seed=42))
    config_file = tmp_path/"tiny.yaml"
    config_file.write_text(yaml.safe_dump(config_value))
    config = load_config(config_file)
    file = tmp_path/"tokens.bin"
    (np.arange(33, dtype="uint16") % 64).tofile(file)
    base = MemmapTokenDataset(file, dtype="uint16", context_length=4)
    view = IndexedTokenWindowDataset(base, np.array([0, 1, 2, 4, 5, 7]), vocab_size=64)
    def loader():
        return create_token_dataloader(view, batch_size=1, shuffle=True, seed=42)
    set_deterministic_seed(42)
    model = LanguageModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
    store = RecoveryStore(tmp_path/"recovery")
    def checkpoint(state, metrics):
        if state.optimizer_step == 1:
            payload = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None,
                state=state, scheduler=SchedulerState(0, 3, .001, .0001), config=config,
                run_name="tiny-indexed", run_id="test", tokenizer_reference=None,
                tokenizer_sha256="tiny", resume_context=dict(data_selection_sha256="selection",
                                                              configuration_sha256="config"))
            store.save(payload)
    options = dict(device=torch.device("cpu"), precision=resolve_precision("fp32", torch.device("cpu")),
        max_steps=3, gradient_accumulation_steps=2, gradient_clip=1., warmup_steps=0,
        peak_learning_rate=.001, minimum_learning_rate=.0001, seed=42,
        loss_normalization="supervised_tokens")
    state, _ = train_model(model, loader(), optimizer, on_optimizer_step=checkpoint, **options)
    restored_model = LanguageModel(config.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=.001, foreach=False)
    restored = load_checkpoint(store.resolve(), model=restored_model, optimizer=restored_optimizer,
                               scaler=None, expected_config=config)
    assert restored.state.optimizer_step == 1 and restored.state.batches_in_epoch == 2
    assert restored.resume_context["data_selection_sha256"] == "selection"
    assert restored.scheduler.minimum_learning_rate == .0001
    resumed, _ = train_model(restored_model, loader(), restored_optimizer, state=restored.state, **options)
    assert resumed.to_dict() == state.to_dict()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            other = restored_optimizer.state_dict()["state"][key][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other
