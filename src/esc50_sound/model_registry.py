"""Local, content-addressed ESC-50 research model registry (schema 1).

Scope is *research-only*. Artifact and chain hashes detect accidental or ordinary
unauthorized modification, but do not authenticate actors or defend against an
attacker who can rewrite the entire registry. Review and release are explicit
human actions. No experiment metrics or real-world safety claims are inferred.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
import shutil

from .inference import AudioClassifier
from .records import canonical_hash, sha256, write_json_new

MAX_FILES = 80
MAX_TOTAL_BYTES = 256 * 1024 * 1024
STEPS = {"candidate": "validated", "validated": "approved", "approved": "released", "released": "archived"}
TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _token(value: str) -> str:
    if not isinstance(value, str) or not TOKEN.fullmatch(value) or value in (".", ".."):
        raise ValueError("Model id and version must be safe short tokens")
    return value


def _file_name(name: str) -> str:
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError("Unsafe artifact filename")
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in ('..', '.', '') for x in p.parts) or ':' in name:
        raise ValueError("Unsafe artifact path")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


@contextmanager
def _lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / '.registry.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return result


def _put_object(root: Path, contents: bytes) -> str:
    digest = _hash_bytes(contents)
    folder = root / 'objects' / digest[:2]
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / digest
    if dst.exists():
        if dst.is_symlink() or sha256(dst) != digest:
            raise ValueError(f"Existing content-addressed object was altered: {digest}")
        return digest
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.object-', delete=False) as f:
            temporary = Path(f.name)
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, dst)  # never replace existing hash-addressed content
        except FileExistsError:
            if dst.is_symlink() or sha256(dst) != digest:
                raise ValueError("Concurrent object publication did not match content hash")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return digest


def _object(root: Path, digest: str, expected_size: int | None = None) -> bytes:
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError("Invalid artifact hash")
    path = root / 'objects' / digest[:2] / digest
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or redirected artifact {digest}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"Artifact length mismatch: {digest}")
    raw = path.read_bytes()
    if _hash_bytes(raw) != digest:
        raise ValueError(f"Artifact hash mismatch: {digest}")
    return raw


def _base(root: Path, model_id: str, version: str) -> Path:
    return root / 'models' / _token(model_id) / _token(version)


def _events(base: Path) -> list[dict]:
    events = sorted((base / 'events').glob('*.json'))
    previous = None
    current = None
    output = []
    for seq, path in enumerate(events, 1):
        if path.name != f'{seq:06d}.json':
            raise ValueError("Missing or reordered lifecycle event")
        event = _read_json(path)
        if event.get('seq') != seq or event.get('previous_hash') != previous:
            raise ValueError("Broken lifecycle event chain")
        if event.get('from') != current or event.get('to') != ("candidate" if seq == 1 else STEPS.get(current)):
            raise ValueError("Illegal lifecycle transition")
        if event.get('scope') != 'research_only':
            raise ValueError("Scope was expanded beyond research-only")
        previous = canonical_hash(event)
        current = event['to']
        output.append(event)
    if not output:
        raise ValueError("Missing candidate registration event")
    return output


def _append(base: Path, transition: str, actor: str, reason: str, *, details=None) -> dict:
    actor = str(actor).strip()
    reason = str(reason).strip()
    if not actor or not reason:
        raise ValueError("Actor and rationale are required")
    previous_events = _events(base)
    before = previous_events[-1]['to']
    if STEPS.get(before) != transition:
        raise ValueError(f"Cannot transition {before} -> {transition}")
    ev = dict(seq=len(previous_events) + 1, at=_now(), **{'from': before, 'to': transition},
              previous_hash=canonical_hash(previous_events[-1]), actor=actor, reason=reason,
              details=details or {}, scope='research_only')
    write_json_new(base / 'events' / f'{ev["seq"]:06d}.json', ev)
    return ev


def _verify_source_bundle(bundle: Path) -> tuple[dict, dict[str, bytes]]:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Bundle must be an ordinary directory")
    manifest_path = bundle / 'bundle_manifest.json'
    if manifest_path.is_symlink():
        raise ValueError("Manifest symlinks are forbidden")
    info = _read_json(manifest_path)
    files = info.get('files')
    if info.get('bundle_schema_version') != 2 or not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Expected an ESC-50 v2 portable bundle manifest")
    if not {'checkpoint.pt', 'MODEL_CARD.md', 'predict.py'}.issubset(files):
        raise ValueError("A release needs a checkpoint, model card and prediction entrypoint")
    raw_files: dict[str, bytes] = {}
    count = 0
    for name, spec in files.items():
        _file_name(name)
        if not isinstance(spec, dict) or not isinstance(spec.get('bytes'), int) or spec['bytes'] < 0 or not SHA.fullmatch(str(spec.get('sha256', ''))):
            raise ValueError(f"Invalid file record: {name}")
        candidate = bundle / name
        if (not candidate.resolve().is_relative_to(bundle.resolve()) or
                any(part.is_symlink() for part in (bundle / PurePosixPath(*PurePosixPath(name).parts[:k])
                                                    for k in range(1, len(PurePosixPath(name).parts) + 1))) or
                not candidate.is_file()):
            raise ValueError(f"Unsafe or missing bundle file: {name}")
        if candidate.stat().st_size != spec['bytes']:
            raise ValueError(f"Bundle file length mismatch: {name}")
        contents = candidate.read_bytes()
        count += len(contents)
        if count > MAX_TOTAL_BYTES or _hash_bytes(contents) != spec['sha256']:
            raise ValueError(f"Bundle file hash or budget failure: {name}")
        raw_files[name] = contents
    # The original manifest is captured unmodified as an additional object.
    raw_files['bundle_manifest.json'] = manifest_path.read_bytes()
    if not raw_files['MODEL_CARD.md'].strip():
        raise ValueError("Model card must not be empty")
    return info, raw_files


def register(root, model_id, version, bundle, *, actor, reason):
    root = Path(root)
    base = _base(root, model_id, version)
    bundle = Path(bundle)
    with _lock(root):
        if base.exists():
            raise FileExistsError("Registry versions cannot be overwritten")
        info, files = _verify_source_bundle(bundle)
        # Validate actual model loading, not just checksum arithmetic.
        classifier = AudioClassifier(bundle / 'checkpoint.pt', device='cpu')
        if info['model'] != classifier.model_name or info.get('model_manifest_id') != classifier.manifest_id:
            raise ValueError("Bundle and embedded checkpoint disagree about model provenance")
        if info.get('provenance_verification') != classifier.verification:
            raise ValueError("Bundle checkpoint provenance verification differs")
        objects = {name: dict(sha256=_put_object(root, raw), bytes=len(raw)) for name, raw in sorted(files.items())}
        manifest = dict(schema_version=1, model_id=_token(model_id), version=_token(version),
                        scope='research_only', model_name=info['model'],
                        source_model_manifest_id=classifier.manifest_id,
                        source_policy=classifier.source_policy,
                        files=objects)
        # Fail-closed if registry's version metadata already exists.
        first = dict(seq=1, at=_now(), **{'from': None, 'to': 'candidate'}, previous_hash=None,
                     actor=str(actor).strip(), reason=str(reason).strip(), details={}, scope='research_only')
        if not first['actor'] or not first['reason']:
            raise ValueError("Actor and rationale required")
        # Publish a complete version directory in one atomic rename. Objects may
        # remain unreferenced after crashes; that is safe and can be GC'd later.
        base.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{version}.staging-', dir=base.parent))
        try:
            write_json_new(temporary / 'manifest.json', manifest)
            write_json_new(temporary / 'events' / '000001.json', first)
            os.rename(temporary, base)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return verify(root, model_id, version, check_release=False)


def verify(root, model_id, version, *, check_release=True):
    root = Path(root)
    base = _base(root, model_id, version)
    manifest = _read_json(base / 'manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('model_id') != model_id or
            manifest.get('version') != version or manifest.get('scope') != 'research_only'):
        raise ValueError("Registry manifest identity or scope mismatch")
    events = _events(base)
    files = manifest.get('files')
    if not isinstance(files, dict) or 'bundle_manifest.json' not in files:
        raise ValueError("Registry artifact manifest incomplete")
    for name, value in files.items():
        _file_name(name)
        _object(root, value['sha256'], value['bytes'])
    original = json.loads(_object(root, files['bundle_manifest.json']['sha256']).decode())
    if set(files) != set(original['files']) | {'bundle_manifest.json'}:
        raise ValueError("Source bundle file set was changed")
    for name, value in original['files'].items():
        if files[name] != value:
            raise ValueError(f"Stored file manifest differs from original: {name}")
    latest = events[-1]['to']
    # Validate full chain and review evidence on every read, not only during transition.
    for ev in events:
        if ev['to'] == 'approved':
            d = ev['details']
            if not d.get('review_sha256') or not d.get('evaluation_sha256') or not d.get('reviewer'):
                raise ValueError('Approval or evaluation evidence is missing')
            _object(root, d['review_sha256'])
            audit = json.loads(_object(root, d['evaluation_sha256']).decode('utf-8'))
            if audit.get('valid') is not True or audit.get('model') != manifest['model_name']:
                raise ValueError('Stored research audit no longer matches the registered model')
        if ev['to'] == 'released':
            d = ev['details']
            archive = root / 'releases' / model_id / (version + '.zip')
            if check_release and (archive.is_symlink() or not archive.is_file() or sha256(archive) != d.get('archive_sha256')):
                raise ValueError('Released archive is missing or modified')
    return dict(model_id=model_id, version=version, model=manifest['model_name'],
                status=latest, scope='research_only', manifest_sha256=sha256(base / 'manifest.json'),
                file_count=len(files), event_count=len(events), checks='pass',
                release_archive=str(root / 'releases' / model_id / (version + '.zip')) if any(e['to'] == 'released' for e in events) else None)


def validate(root, model_id, version, *, actor, reason):
    root = Path(root)
    with _lock(root):
        report = verify(root, model_id, version)
        if report['status'] != 'candidate':
            raise ValueError('Expected candidate state')
        _append(_base(root, model_id, version), 'validated', actor, reason,
                details={'technical_check': 'bundle hashes, embedded provenance, artifact integrity'})
        return verify(root, model_id, version)


def approve(root, model_id, version, *, reviewer, review_document, evaluation_report, reason):
    """Human attestation for *research* release; does NOT authorize production."""
    root = Path(root)
    with _lock(root):
        report = verify(root, model_id, version)
        if report['status'] != 'validated':
            raise ValueError('Only technically validated models can be reviewed')
        source = Path(review_document)
        if source.is_symlink() or not source.is_file() or not 30 <= source.stat().st_size <= 128_000:
            raise ValueError('Provide a nonempty, bounded review document')
        contents = source.read_bytes()
        evaluation_source = Path(evaluation_report)
        if (evaluation_source.is_symlink() or not evaluation_source.is_file() or
                not 20 <= evaluation_source.stat().st_size <= 2_000_000):
            raise ValueError('A bounded audited evaluation report is required')
        audit_bytes = evaluation_source.read_bytes()
        audit = json.loads(audit_bytes.decode('utf-8'))
        manifest = _read_json(_base(root, model_id, version) / 'manifest.json')
        if (not isinstance(audit, dict) or audit.get('schema_version') != 1 or
                audit.get('valid') is not True or audit.get('model') != manifest['model_name'] or
                not audit.get('selection_disclosure')):
            raise ValueError('Evaluation audit is not valid, matching, or candid about selection history')
        reviewer = str(reviewer).strip()
        if not reviewer:
            raise ValueError('Reviewer is required')
        digest = _put_object(root, contents)
        audit_digest = _put_object(root, audit_bytes)
        _append(_base(root, model_id, version), 'approved', reviewer, reason,
                details={'review_sha256': digest, 'evaluation_sha256': audit_digest,
                         'evaluation_scope': audit.get('assessment'), 'reviewer': reviewer,
                         'usage_scope': 'noncommercial_research_only',
                         'scientific_confirmation': 'not established by registry'})
        return verify(root, model_id, version)


def _archive_bytes(root: Path, model_id: str, version: str) -> bytes:
    base = _base(root, model_id, version)
    manifest = _read_json(base / 'manifest.json')
    files = manifest['files']
    release_record = dict(release_schema_version=1, registry_model_id=model_id, version=version,
                          scope='research_only', registry_manifest_sha256=sha256(base / 'manifest.json'),
                          warning='Research-only. No independent confirmation, calibration, or production authorization is implied.')
    contents = {n: _object(root, spec['sha256'], spec['bytes']) for n, spec in files.items()}
    contents['REGISTRY_RELEASE.json'] = (json.dumps(release_record, indent=2) + '\n').encode()
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(contents.items()):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, raw)
    return data.getvalue()


def release(root, model_id, version, *, actor, reason):
    root = Path(root)
    with _lock(root):
        report = verify(root, model_id, version)
        if report['status'] != 'approved':
            raise ValueError('Explicit reviewer approval required before research release')
        raw = _archive_bytes(root, model_id, version)
        path = root / 'releases' / model_id / (version + '.zip')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256(path) != _hash_bytes(raw):
                raise ValueError('Conflicting orphan release archive from interrupted operation')
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as f:
                    temporary = Path(f.name)
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                os.link(temporary, path)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
        _append(_base(root, model_id, version), 'released', actor, reason,
                details={'archive_sha256': _hash_bytes(raw), 'scope': 'research_only'})
        return verify(root, model_id, version)


def archive(root, model_id, version, *, actor, reason):
    root = Path(root)
    with _lock(root):
        report = verify(root, model_id, version)
        if report['status'] != 'released':
            raise ValueError('Only released models can be archived')
        current = _channel_events(root, 'research')
        if current and current[-1]['to'] == {'model_id': model_id, 'version': version}:
            raise ValueError('Rollback or promote another release before archiving an active version')
        _append(_base(root, model_id, version), 'archived', actor, reason)
        return verify(root, model_id, version)


def _channel_events(root: Path, channel: str) -> list[dict]:
    if channel != 'research':
        raise ValueError('Only the research channel is implemented')
    folder = root / 'channels' / channel
    entries = sorted(folder.glob('*.json'))
    previous_hash = None
    current = None
    out = []
    for idx, path in enumerate(entries, 1):
        if path.name != f'{idx:06d}.json':
            raise ValueError('Missing channel history')
        ev = _read_json(path)
        if ev.get('seq') != idx or ev.get('previous_hash') != previous_hash or ev.get('from') != current:
            raise ValueError('Channel history changed or reordered')
        previous_hash = canonical_hash(ev)
        current = ev.get('to')
        out.append(ev)
    return out


def set_research_channel(root, model_id, version, *, actor, reason, rollback=False):
    root = Path(root)
    with _lock(root):
        report = verify(root, model_id, version)
        if report['status'] != 'released':
            raise ValueError('Only released, nonarchived model versions can be activated')
        previous = _channel_events(root, 'research')
        target = dict(model_id=model_id, version=version)
        current = previous[-1]['to'] if previous else None
        if target == current:
            raise ValueError('Research channel already points to this version')
        if rollback and target not in [ev['to'] for ev in previous[:-1]]:
            raise ValueError('Rollback target must be a previously active released version')
        if not str(actor).strip() or not str(reason).strip():
            raise ValueError('Actor and rationale are required')
        ev = dict(seq=len(previous)+1, at=_now(), **{'from': current, 'to': target},
                  previous_hash=canonical_hash(previous[-1]) if previous else None,
                  actor=str(actor).strip(), reason=str(reason).strip(),
                  action='rollback' if rollback else 'promote', scope='research_only')
        write_json_new(root / 'channels' / 'research' / f'{ev["seq"]:06d}.json', ev)
        return dict(channel='research', active=target, previous=current, action=ev['action'])


def resolve_research_channel(root):
    root = Path(root)
    events = _channel_events(root, 'research')
    if not events:
        return dict(channel='research', active=None)
    target = events[-1]['to']
    report = verify(root, target['model_id'], target['version'])
    if report['status'] != 'released':
        raise ValueError('Active channel points to nonreleased model')
    return dict(channel='research', active=target, archive=report['release_archive'],
                history_length=len(events), scope='research_only')


def list_models(root):
    """List all registered versions; verify integrity before displaying a status."""
    root = Path(root)
    folder = root / 'models'
    if not folder.exists():
        return []
    rows = []
    for model in sorted(folder.iterdir()):
        if model.name.startswith('.'):
            continue
        if model.is_symlink() or not model.is_dir():
            raise ValueError('Invalid registry model folder')
        for version in sorted(model.iterdir()):
            if version.name.startswith('.'):
                continue
            if version.is_symlink() or not version.is_dir():
                raise ValueError('Invalid registry version folder')
            rows.append(verify(root, model.name, version.name))
    return rows
