"""Phase 4: research-only model registry invariants and real portable bundles."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile

import pytest

from esc50_sound import model_registry as reg

BASE = Path(__file__).resolve().parents[1]
BUNDLE_CNN = BASE / 'models/cnn'
BUNDLE_CRNN = BASE / 'models/crnn'


def add(registry, version='v1', bundle=BUNDLE_CNN):
    return reg.register(registry, 'esc50', version, bundle,
                        actor='test-reviewer', reason='Register existing historical research package')


def pass_review(registry, version, tmp_path):
    reg.validate(registry, 'esc50', version, actor='ci', reason='Technical checks passed')
    note = tmp_path / f'review-{version}.md'
    note.write_text('Research only. Holdout has already been accessed. No field or production readiness is established.\n')
    report = (BASE / 'results/historical/infra_v3/crnn_freq_cv_seed101_audit.json' if version == 'v2'
              else BASE / 'results/historical/infra_v3/cnn_freq_cv_seed101_audit.json')
    reg.approve(registry, 'esc50', version, reviewer='human-reviewer', review_document=note, evaluation_report=report,
                reason='Review permits archived research use only')
    return reg.release(registry, 'esc50', version, actor='human-reviewer', reason='Publish research-only version')


def test_register_candidate_and_validation(tmp_path):
    root = tmp_path / 'registry'
    a = add(root)
    assert a['status'] == 'candidate' and a['file_count'] == 19
    with pytest.raises(FileExistsError):
        add(root)
    with pytest.raises(ValueError, match='Explicit reviewer approval'):
        reg.release(root, 'esc50', 'v1', actor='a', reason='no')
    b = reg.validate(root, 'esc50', 'v1', actor='ci', reason='Technical audit passed')
    assert b['status'] == 'validated'
    with pytest.raises(ValueError):
        reg.validate(root, 'esc50', 'v1', actor='ci', reason='repeat')
    with pytest.raises(ValueError, match='review document'):
        reg.approve(root, 'esc50', 'v1', reviewer='human', review_document=tmp_path / 'missing.md', evaluation_report=BASE / 'results/historical/infra_v3/cnn_freq_cv_seed101_audit.json', reason='no')


def test_research_release_and_verify(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    released = pass_review(root, 'v1', tmp_path)
    assert released['status'] == 'released'
    path = Path(released['release_archive'])
    with zipfile.ZipFile(path) as f:
        assert set(f.namelist()) >= {'checkpoint.pt', 'bundle_manifest.json', 'MODEL_CARD.md', 'REGISTRY_RELEASE.json'}
        rec = json.loads(f.read('REGISTRY_RELEASE.json'))
        assert rec['scope'] == 'research_only'
        assert rec['registry_model_id'] == 'esc50'
        original = json.loads(f.read('bundle_manifest.json'))
        for name, spec in original['files'].items():
            assert reg._hash_bytes(f.read(name)) == spec['sha256']
    assert reg.verify(root, 'esc50', 'v1')['checks'] == 'pass'
    with pytest.raises(ValueError):
        reg.release(root, 'esc50', 'v1', actor='x', reason='duplicate')


def test_tampered_object_detected(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    m = reg._read_json(root / 'models/esc50/v1/manifest.json')
    file_entry = m['files']['MODEL_CARD.md']
    path = root / 'objects' / file_entry['sha256'][:2] / file_entry['sha256']
    path.write_bytes(b'invalid replacement')
    with pytest.raises(ValueError, match='length mismatch|hash mismatch'):
        reg.verify(root, 'esc50', 'v1')
    with pytest.raises(ValueError):
        reg.validate(root, 'esc50', 'v1', actor='ci', reason='not acceptable')


def test_event_chain_mutation_detected(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    reg.validate(root, 'esc50', 'v1', actor='ci', reason='test')
    ev = root / 'models/esc50/v1/events/000001.json'
    payload = json.loads(ev.read_text())
    payload['actor'] = 'forged-reviewer'
    ev.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Broken lifecycle event chain'):
        reg.verify(root, 'esc50', 'v1')


def test_release_corruption_detected(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    report = pass_review(root, 'v1', tmp_path)
    Path(report['release_archive']).write_bytes(b'not a zip')
    with pytest.raises(ValueError, match='missing or modified'):
        reg.verify(root, 'esc50', 'v1')
    with pytest.raises(ValueError):
        reg.set_research_channel(root, 'esc50', 'v1', actor='a', reason='invalid')


def test_research_channel_rollback(tmp_path):
    root = tmp_path / 'registry'
    add(root, 'v1', BUNDLE_CNN)
    pass_review(root, 'v1', tmp_path)
    add(root, 'v2', BUNDLE_CRNN)
    pass_review(root, 'v2', tmp_path)
    reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='Initial research model')
    reg.set_research_channel(root, 'esc50', 'v2', actor='human', reason='Updated research model')
    active = reg.resolve_research_channel(root)
    assert active['active']['version'] == 'v2'
    reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='Return to previous', rollback=True)
    assert reg.resolve_research_channel(root)['active']['version'] == 'v1'
    # A previous release can be restored again; only never-active targets are rejected.
    back = reg.set_research_channel(root, 'esc50', 'v2', actor='human', reason='Restore prior version', rollback=True)
    assert back['active']['version'] == 'v2'


def test_no_unreleased_promotion_or_production_channel(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    with pytest.raises(ValueError, match='Only released'):
        reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='unsafe')
    with pytest.raises(ValueError, match='Only the research'):
        reg._channel_events(root, 'production')


def test_bad_bundle_fail_closed(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'bundle_manifest.json').write_text('{"bundle_schema_version":2,"files": {"../escape": {"sha256": "' + '0'*64 + '","bytes": 1}}}')
    with pytest.raises(ValueError):
        reg.register(tmp_path / 'r', 'model', 'v1', bundle, actor='a', reason='b')
    assert not (tmp_path / 'r/models/model/v1/manifest.json').exists()


def test_versions_and_identity_sanitization(tmp_path):
    with pytest.raises(ValueError):
        reg.register(tmp_path, '../../escape', 'v1', BUNDLE_CNN, actor='a', reason='b')
    with pytest.raises(ValueError):
        reg.register(tmp_path, 'model', '../bad', BUNDLE_CNN, actor='a', reason='b')


def test_reviewer_does_not_claim_independent_confirmation(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    pass_review(root, 'v1', tmp_path)
    events = reg._events(root / 'models/esc50/v1')
    approval = next(e for e in events if e['to']=='approved')
    assert approval['details']['scientific_confirmation'] == 'not established by registry'
    assert approval['details']['usage_scope'] == 'noncommercial_research_only'
    assert approval['details']['evaluation_sha256']


def test_invalid_evaluation_evidence_blocked(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    reg.validate(root, 'esc50', 'v1', actor='ci', reason='test')
    note = tmp_path / 'review.md'
    note.write_text('Research only, no independent confirmation and not suitable for real-world alarms.')
    wrong = BASE / 'results/historical/infra_v3/crnn_freq_cv_seed101_audit.json'
    with pytest.raises(ValueError, match='not valid, matching'):
        reg.approve(root, 'esc50', 'v1', reviewer='r', review_document=note,
                    evaluation_report=wrong, reason='wrong model')
    forged = tmp_path / 'forged.json'
    forged.write_text(json.dumps({'schema_version':1,'valid':False,'model':'cnn_freq',
                                  'selection_disclosure':'Previous test access was not independent.'}))
    with pytest.raises(ValueError, match='not valid, matching'):
        reg.approve(root, 'esc50', 'v1', reviewer='r', review_document=note,
                    evaluation_report=forged, reason='invalid status')
    assert reg.verify(root, 'esc50', 'v1')['status'] == 'validated'


def test_current_release_cannot_be_archived(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    pass_review(root, 'v1', tmp_path)
    reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='research default')
    with pytest.raises(ValueError, match='before archiving'):
        reg.archive(root, 'esc50', 'v1', actor='human', reason='should not permit')
    assert reg.verify(root, 'esc50', 'v1')['status'] == 'released'


def test_nested_symlinks_rejected(tmp_path):
    source = BUNDLE_CNN
    bundle = tmp_path / 'copied'
    import shutil
    shutil.copytree(source, bundle)
    folder = bundle / 'src'
    shutil.rmtree(folder)
    folder.symlink_to(source / 'src', target_is_directory=True)
    with pytest.raises(ValueError, match='Unsafe or missing'):
        reg.register(tmp_path / 'registry', 'test', 'v1', bundle, actor='user', reason='unsafe path')


def test_channel_event_chain_tamper_detected(tmp_path):
    root = tmp_path / 'registry'
    add(root, 'v1', BUNDLE_CNN)
    pass_review(root, 'v1', tmp_path)
    add(root, 'v2', BUNDLE_CRNN)
    pass_review(root, 'v2', tmp_path)
    reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='first')
    reg.set_research_channel(root, 'esc50', 'v2', actor='human', reason='second')
    history = root / 'channels/research/000001.json'
    record = json.loads(history.read_text())
    record['to']['version'] = 'fake'
    history.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Channel history changed'):
        reg.resolve_research_channel(root)


def test_list_all_registry_versions(tmp_path):
    root = tmp_path / 'registry'
    assert reg.list_models(root) == []
    add(root, 'v1', BUNDLE_CNN)
    add(root, 'v2', BUNDLE_CRNN)
    listed = reg.list_models(root)
    assert len(listed) == 2 and [x['version'] for x in listed] == ['v1', 'v2']
    assert all(x['status'] == 'candidate' for x in listed)


def test_archive_unselected_research_release(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    pass_review(root, 'v1', tmp_path)
    archived = reg.archive(root, 'esc50', 'v1', actor='human', reason='Research version retired')
    assert archived['status'] == 'archived'
    assert reg.verify(root, 'esc50', 'v1')['checks'] == 'pass'
    with pytest.raises(ValueError, match='Only released'):
        reg.set_research_channel(root, 'esc50', 'v1', actor='human', reason='retired')


def test_approval_evidence_alteration_detected(tmp_path):
    root = tmp_path / 'registry'
    add(root)
    pass_review(root, 'v1', tmp_path)
    events = reg._events(root / 'models/esc50/v1')
    approval = next(e for e in events if e['to'] == 'approved')
    digest = approval['details']['evaluation_sha256']
    obj = root / 'objects' / digest[:2] / digest
    obj.write_bytes(b'{"valid": true}')
    with pytest.raises(ValueError):
        reg.verify(root, 'esc50', 'v1')
