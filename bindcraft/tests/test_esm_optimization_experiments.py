import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmark/script/run_esm_optimization.py"
spec = importlib.util.spec_from_file_location("esm_optimization_experiments", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_candidate_selection_is_unique_and_deterministic(tmp_path):
    directory = tmp_path / "esmfold2_artifacts"
    directory.mkdir()
    rows = [
        {"sequence": "AAAA", "total_loss": 3},
        {"sequence": "CCCC", "total_loss": 2},
        {"sequence": "AAAA", "total_loss": 1},
        {"sequence": "DDDD", "total_loss": 2},
    ]
    (directory / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    selected, unique, calls = experiment.selected_candidates(tmp_path, 2)
    assert [row["sequence"] for row in selected] == ["AAAA", "CCCC"]
    assert unique == 3 and calls == 4


def test_cluster_bootstrap_does_not_overweight_hotspot_variants():
    result = experiment.clustered_interval({"target1": [1] * 100, "target2": [3]})
    assert result["mean"] == 2
    assert result["source_targets"] == 2
    assert result["interpretation"] == "exploratory"
    assert result == experiment.clustered_interval({"target1": [1] * 100, "target2": [3]})


def test_prepare_uses_disjoint_outputs_and_common_seeds(tmp_path):
    args = experiment.parser().parse_args([
        "prepare", "--output", str(tmp_path), "--kit-root", str(tmp_path / "kit"),
        "--cases", "denovo_PD1", "--arms", "baseline", "exact", "--seeds", "43", "44",
    ])
    experiment.prepare(args)
    manifest = json.loads((tmp_path / "campaign.json").read_text())
    rows = manifest["records"]
    assert len(rows) == 4
    assert len({row["directory"] for row in rows}) == 4
    for row in rows:
        command = row["command"]
        assert command[command.index("--seed") + 1] == str(row["seed"])
        assert command[command.index("--esmfold2_save_verified_candidate_cif") + 1] == "true"
        assert "--esmfold2_optimization_timing" in command
    experiment.prepare(args)
    assert json.loads((tmp_path / "campaign.json").read_text()) == manifest


def test_prepare_refuses_override_of_arm_factor(tmp_path):
    args = experiment.parser().parse_args([
        "prepare", "--output", str(tmp_path), "--kit-root", str(tmp_path),
        "--cases", "denovo_PD1", "--set", "esmfold2_esmc_optimization=exact",
    ])
    with pytest.raises(ValueError, match="experimental factors"):
        experiment.prepare(args)


def test_report_preserves_missing_quality_and_failures(tmp_path):
    records = []
    for arm, code in (("baseline", 0), ("exact", 1)):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "experiment_run.json").write_text(json.dumps({"returncode": code, "wall_s": 10}))
        records.append({"arm": arm, "seed": 43, "case": {"case_id": "x", "target_name": "source"}, "directory": str(directory)})
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"records": records}))
    output = tmp_path / "report.csv"
    experiment.report(SimpleNamespace(manifest=str(manifest), scores=str(tmp_path / "absent"), output=str(output)))
    report = json.loads(output.with_suffix(".summary.json").read_text())
    assert report["exact"]["failed"] == 1
    assert report["exact"]["paired_quality_delta"] is None
    assert report["exact"]["paired_speedup"] is None


def test_comparison_handles_constrained_infinities_and_rejects_nan():
    import numpy as np
    a = np.array([1.0, -np.inf, np.inf, 0.0])
    result = experiment.compare_tensors(a, a.copy())
    assert result['bitwise_equal'] and result['nonfinite_pattern_equal']
    assert result['max_abs'] == result['relative_l2'] == 0
    b = a.copy()
    b[0] = 2
    assert experiment.compare_tensors(a, b)['max_abs'] == 1
    b[1] = np.inf
    assert not experiment.compare_tensors(a, b)['nonfinite_pattern_equal']
    assert experiment.compare_tensors(a, b)['relative_l2'] is None
    with pytest.raises(ValueError, match='NaN'):
        experiment.compare_tensors(np.array([np.nan]), np.array([np.nan]))


def test_verifier_counts_use_runtime_calls_not_candidate_rows(tmp_path):
    assert experiment.verifier_counts(tmp_path)['verifier_calls'] is None
    (tmp_path / 'optimization_report.json').write_text(json.dumps({
        'counters': {'verifier.calls': 2, 'verifier.successes': 1, 'verifier.errors': 1}}))
    assert experiment.verifier_counts(tmp_path) == {
        'verifier_calls': 2, 'verifier_successes': 1, 'verifier_errors': 1}


def rescore_fixture(tmp_path):
    directory = tmp_path / 'search'
    directory.mkdir()
    (directory / 'experiment_run.json').write_text(json.dumps({'returncode': 0}))
    artifacts = directory / 'esmfold2_artifacts'
    artifacts.mkdir()
    (artifacts / 'candidates.jsonl').write_text(json.dumps({'sequence': 'AAAA', 'total_loss': 1}))
    campaign = {'records': [{'arm': 'baseline', 'seed': 43,
        'case': {'case_id': 'x', 'target_name': 'target'}, 'directory': str(directory)}]}
    manifest = tmp_path / 'campaign.json'
    experiment.write_json(manifest, campaign)
    args = SimpleNamespace(manifest=manifest, top_k=5, seeds=[1043, 1044], loops=20,
                           sampling_steps=200, verifier_model='pinned-model')
    spec = experiment.freeze_rescore_spec(args, campaign)
    root = tmp_path / 'scores'
    experiment.write_json(root / 'rescore_spec.json', spec)
    cif = 'data_test\n#\n'
    for name in ('one', 'two', 'duplicate'):
        directory = root / name
        directory.mkdir()
        (directory / 'complex.cif').write_text(cif)
    scores = [{**item, **{k: spec[k] for k in ('contract_hash', 'loops', 'sampling_steps', 'model', 'evaluator')},
               'structure_sha256': hashlib.sha256(cif.encode()).hexdigest(),
               'metrics': {'pair_chain_iptm': 0.8}} for item in spec['expected']]
    return root, manifest, spec, scores


def test_incomplete_rescore_matrix_cannot_contribute_quality(tmp_path):
    root, manifest, spec, scores = rescore_fixture(tmp_path)
    experiment.write_json(root / 'one/score.json', scores[0])
    valid, required, observed = experiment.validated_scores(root, manifest)
    assert valid == []
    assert required[('baseline', 'x', 43)] == 2
    assert observed[('baseline', 'x', 43)] == 1
    experiment.write_json(root / 'two/score.json', scores[1])
    assert len(experiment.validated_scores(root, manifest)[0]) == 2


@pytest.mark.parametrize('fault', ['duplicate', 'sequence', 'contract', 'nan',
                                  'missing_structure', 'changed_structure', 'legacy_score'])
def test_rescore_rejects_invalid_records(tmp_path, fault):
    root, manifest, spec, scores = rescore_fixture(tmp_path)
    if fault == 'duplicate':
        experiment.write_json(root / 'duplicate/score.json', scores[0])
    elif fault == 'sequence':
        scores[0]['binder_seq'] = 'GGGG'
    elif fault == 'contract':
        scores[0]['contract_hash'] = 'wrong'
    elif fault == 'missing_structure':
        (root / 'one/complex.cif').unlink()
    elif fault == 'changed_structure':
        (root / 'one/complex.cif').write_text('corrupted')
    elif fault == 'legacy_score':
        scores[0].pop('structure_sha256')
    else:
        scores[0]['metrics']['pair_chain_iptm'] = float('nan')
    experiment.write_json(root / 'one/score.json', scores[0])
    with pytest.raises(ValueError):
        experiment.validated_scores(root, manifest)


def test_failed_search_is_retained_without_blocking_other_rescores(tmp_path):
    root, manifest, spec, scores = rescore_fixture(tmp_path)
    campaign = json.loads(manifest.read_text())
    failed = tmp_path / 'failed'
    failed.mkdir()
    experiment.write_json(failed / 'experiment_run.json', {'returncode': 1})
    campaign['records'].append({'arm': 'fast', 'seed': 43,
        'case': {'case_id': 'x', 'target_name': 'target'}, 'directory': str(failed)})
    experiment.write_json(manifest, campaign)
    args = SimpleNamespace(manifest=manifest, top_k=5, seeds=[1043, 1044], loops=20,
                           sampling_steps=200, verifier_model='pinned-model')
    frozen = experiment.freeze_rescore_spec(args, campaign)
    assert len(frozen['expected']) == 2
    assert frozen['failed_searches'] == [{'arm': 'fast', 'case_id': 'x',
                                         'search_seed': 43, 'returncode': 1}]
    experiment.write_json(failed / 'experiment_run.json', {'returncode': None})
    with pytest.raises(ValueError, match='finish'):
        experiment.freeze_rescore_spec(args, campaign)


@pytest.mark.parametrize('interrupt_at', [None, 'structure', 'score'])
def test_campaign_rescore_executes_with_frozen_config_and_resumes(tmp_path, monkeypatch, interrupt_at):
    from bindcraft.hallucination import esmfold2_runtime

    _, manifest, _, _ = rescore_fixture(tmp_path)
    campaign = json.loads(manifest.read_text())
    campaign['records'][0]['case'].update(
        task_type='denovo', target_sequence='CCCC', target_chain_roles=['target'],
        mutable_mask=None, target_hotspots_1based=[1],
    )
    experiment.write_json(manifest, campaign)
    args = SimpleNamespace(
        manifest=manifest, output=tmp_path / 'executed', top_k=5, seeds=[1043, 1044],
        loops=20, sampling_steps=200, verifier_model='pinned-model',
        proposal_model='unused-proposal', kit_root=None, device='cpu',
    )
    calls, configs = [], []

    class Backend:
        def __init__(self, **kwargs):
            configs.append(kwargs['config'])
            self.verifier_model = object()
            self.input_builder = object()
            self.optimizations = SimpleNamespace(finalize=lambda: None)

        def verify(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(total_loss=0.2, loss_dict={}, summary_confidence={
                'pair_chain_iptm': 0.8, 'mmcif': 'data_test\n#\n',
            })

    monkeypatch.setattr(esmfold2_runtime, 'ESMFold2RuntimeBackend', Backend)
    if interrupt_at:
        original_replace = Path.replace
        def interrupt_commit(path, target):
            if Path(target).name == ('complex.cif' if interrupt_at == 'structure' else 'score.json'):
                raise OSError('simulated interrupted write')
            return original_replace(path, target)
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'replace', interrupt_commit)
            with pytest.raises(OSError, match='simulated interrupted write'):
                experiment.rescore(args)
        assert not list(args.output.rglob('score.json'))
        assert experiment.validated_scores(args.output, manifest)[0] == []
    experiment.rescore(args)
    experiment.rescore(args)
    assert [call['seed'] for call in calls] == ([1043] if interrupt_at else []) + [1043, 1044]
    assert all(call['hotspot_params']['target_hotspot_concat_indices'] == [0] for call in calls)
    assert all((cfg.proposal_optimization, cfg.verifier_optimization, cfg.esmc_optimization)
               == ('off', 'off', 'off') for cfg in configs)
    assert configs[0].optimization_report_path.endswith('baseline/x/seed-43/optimization_report.json')
    assert len(experiment.validated_scores(args.output, manifest)[0]) == 2
    assert len(list(args.output.rglob('complex.cif'))) == 2
    structure = next(args.output.rglob('complex.cif'))
    structure.write_text('corrupted')
    with pytest.raises(ValueError, match='structure is missing or changed'):
        experiment.rescore(args)
