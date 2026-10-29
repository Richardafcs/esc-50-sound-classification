"""Verified compact checkpoints and self-contained notebook evidence capsules.

Only explicitly selected local project files enter capsules. No mount, OAuth,
network transfer, secret discovery, or model evaluation is performed here.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

import torch
import esc50 as esc

MAX_CAPSULE_BYTES = 8 * 1024 ** 2
MAX_EXPANDED_BYTES = 64 * 1024 ** 2
MANIFEST_NAME = "EVIDENCE_MANIFEST.json"
TRAINING_KEYS = {"optimizer", "scheduler", "torch_rng", "generator_rng", "cuda_rng",
                 "python_rng", "numpy_rng", "rng_state", "random_state"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _new_file(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_same(x, y) for x, y in zip(left, right))
    return left == right


def compact_checkpoint(source, destination):
    """Discard training state while preserving every inference/provenance field."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {destination}")
    original = esc.load_checkpoint(source)
    esc.validate_checkpoint(original)
    omitted = sorted(TRAINING_KEYS.intersection(original))
    compact = {key: value for key, value in original.items() if key not in TRAINING_KEYS}
    compact["inference_artifact"] = dict(schema_version=1, original_checkpoint_sha256=esc.sha256(source),
                                          omitted_training_fields=omitted)
    esc.validate_checkpoint(compact)
    data = io.BytesIO()
    torch.save(compact, data)
    raw = data.getvalue()
    # Validate serialized bytes before publishing the destination.
    restored = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    esc.validate_checkpoint(restored)
    for key, value in original.items():
        if key not in TRAINING_KEYS and key != "inference_artifact" and not _same(value, restored[key]):
            raise ValueError(f"Checkpoint compaction changed retained field: {key}")
    _new_file(destination, raw)
    return dict(source=str(source), destination=str(destination), original_bytes=source.stat().st_size,
                compact_bytes=len(raw), compact_sha256=_sha(raw), omitted_training_fields=omitted,
                verification="all retained fields and model tensors exactly preserved")


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Evidence names must be nonempty POSIX relative paths")
    path = PurePosixPath(name)
    if (name == "." or path.is_absolute() or name.endswith("/") or any(part in ("..", ".") or ":" in part for part in path.parts)
            or str(path) != name):
        raise ValueError(f"Unsafe evidence path: {name}")
    return path


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("capsule_id must be a short alphanumeric identifier with optional _ or -")
    return value


def create_capsule(root, files, output, *, capsule_id="evidence", max_bytes=MAX_CAPSULE_BYTES):
    """ZIP explicitly listed project files with an internal per-file SHA manifest.

    A group exceeding the cap is rejected; callers must select explicit groups
    themselves, preserving an inspectable boundary between capsule contents.
    """
    root, output = Path(root).resolve(), Path(output)
    capsule_id = _identifier(capsule_id)
    if output.exists():
        raise FileExistsError(f"Refusing to replace capsule: {output}")
    if not isinstance(files, (list, tuple)) or not files:
        raise ValueError("files must be an explicit nonempty list; glob discovery is not supported")
    if not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_CAPSULE_BYTES:
        raise ValueError("Capsule byte cap must be positive and at most 8 MiB")
    contents = {}
    total = 0
    for supplied in files:
        selected = Path(supplied)
        if selected.is_absolute():
            relative = selected.relative_to(root).as_posix()
        else:
            relative = str(supplied)
        _relative(relative)
        if relative == MANIFEST_NAME or relative in contents:
            raise ValueError("Duplicate or reserved capsule filename")
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Selected evidence is not a regular file inside root: {relative}")
        total += path.stat().st_size
        if total > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded capsule exceeds 64 MiB limit; split explicit file groups")
        contents[relative] = path.read_bytes()
    manifest = dict(schema_version=1, capsule_id=capsule_id,
                    files={name: dict(bytes=len(raw), sha256=_sha(raw)) for name, raw in contents.items()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, sort_keys=True, indent=2).encode())
    raw = buffer.getvalue()
    if len(raw) > max_bytes:
        raise ValueError("Compressed capsule exceeds its limit; split explicit file groups")
    validate_capsule(raw)
    _new_file(output, raw)
    return dict(path=str(output), capsule_id=capsule_id, bytes=len(raw), sha256=_sha(raw), files=list(contents))


def validate_capsule(raw, expected_sha256=None):
    """Validate bytes, paths, expansion bounds, membership and every file hash."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_CAPSULE_BYTES:
        raise ValueError("Capsule must contain between one byte and 8 MiB")
    if expected_sha256 is not None and _sha(raw) != expected_sha256:
        raise ValueError("Capsule SHA256 does not match the saved HTML declaration")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) > 2048 or sum(member.file_size for member in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Capsule expansion/member limit exceeded")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP members")
            for member in members:
                _relative(member.filename)
                # Only regular files; no archive symlinks or special devices.
                mode = member.external_attr >> 16
                if member.is_dir() or (mode & 0o170000) not in (0, 0o100000):
                    raise ValueError("Capsule contains a nonregular member")
            if MANIFEST_NAME not in names:
                raise ValueError("Capsule has no evidence manifest")
            manifest = json.loads(archive.read(MANIFEST_NAME))
            if (manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict)
                    or not manifest["files"] or MANIFEST_NAME in manifest["files"]):
                raise ValueError("Unsupported evidence manifest schema")
            _identifier(manifest.get("capsule_id"))
            if set(names) != set(manifest["files"]) | {MANIFEST_NAME}:
                raise ValueError("ZIP member list disagrees with manifest")
            contents = {}
            for name, declared in manifest["files"].items():
                _relative(name)
                content = archive.read(name)
                if not isinstance(declared, dict) or len(content) != declared.get("bytes") or _sha(content) != declared.get("sha256"):
                    raise ValueError(f"Evidence file hash/size mismatch: {name}")
                contents[name] = content
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Malformed evidence capsule") from error
    return manifest, contents


def capsule_html(zip_path, label=None):
    raw = Path(zip_path).read_bytes()
    manifest, _ = validate_capsule(raw)
    digest = _sha(raw)
    filename = manifest["capsule_id"] + ".zip"
    title = label or f"Save evidence capsule {manifest['capsule_id']} ({len(raw):,} bytes)"
    return (f'<a download="{html.escape(filename, quote=True)}" '
            f'data-capsule-id="{manifest["capsule_id"]}" data-evidence-sha256="{digest}" '
            f'href="data:application/zip;base64,{base64.b64encode(raw).decode()}">{html.escape(title)}</a>')


def display_capsule(zip_path, label=None):
    """Store recoverable bytes directly in the notebook's text/html output."""
    from IPython.display import display
    markup = capsule_html(zip_path, label)
    display({"text/html": markup}, raw=True)
    # No HTML return value: avoid a second pretty-printed copy of the base64 bytes.


class _CapsuleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() != "a" or "data-evidence-sha256" not in attrs:
            return
        prefix = "data:application/zip;base64,"
        href = attrs.get("href", "")
        digest = attrs["data-evidence-sha256"]
        if not href.startswith(prefix) or not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
            raise ValueError("Saved capsule anchor has invalid data URL/hash")
        encoded = href[len(prefix):]
        if len(encoded) > 4 * ((MAX_CAPSULE_BYTES + 2) // 3):
            raise ValueError("Saved capsule data URL exceeds limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("Saved capsule base64 is malformed") from error
        manifest, contents = validate_capsule(raw, digest)
        if attrs.get("data-capsule-id", manifest["capsule_id"]) != manifest["capsule_id"]:
            raise ValueError("Saved capsule identifier differs from ZIP manifest")
        self.items.append((raw, digest, manifest, contents))


def extract_capsules(executed_ipynb, output):
    """Recover only capsules actually present in saved notebook output HTML."""
    notebook_path, output = Path(executed_ipynb), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace recovered evidence: {output}")
    raw_notebook = notebook_path.read_bytes()
    notebook = json.loads(raw_notebook)
    recovered = {}
    locations = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        for output_index, item in enumerate(cell.get("outputs", [])):
            markup = item.get("data", {}).get("text/html")
            if markup is None:
                continue
            if isinstance(markup, list):
                markup = "".join(markup)
            if not isinstance(markup, str):
                raise ValueError("Notebook text/html output has invalid type")
            parser = _CapsuleParser()
            parser.feed(markup)
            for raw, digest, manifest, contents in parser.items:
                recovered[digest] = (raw, manifest, contents)
                locations.append(dict(cell=cell_index, output=output_index, sha256=digest))
    if not recovered:
        raise ValueError("No verified evidence capsules were found in saved notebook outputs")
    # All archives are validated before any extraction; no partially trusted tree.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for digest, (raw, manifest, contents) in recovered.items():
        directory = output / f"{manifest['capsule_id']}-{digest[:12]}"
        directory.mkdir()
        _new_file(directory / "capsule.zip", raw)
        for name, content in contents.items():
            _new_file(directory / "files" / name, content)
        _new_file(directory / "manifest.json", json.dumps(manifest, indent=2).encode())
        records.append(dict(capsule_id=manifest["capsule_id"], sha256=digest,
                            directory=str(directory), files=list(contents)))
    report = dict(schema_version=1, notebook_sha256=_sha(raw_notebook), capsules=records, output_locations=locations)
    _new_file(output / "recovery.json", json.dumps(report, indent=2).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compact = sub.add_parser("compact")
    compact.add_argument("--source", required=True)
    compact.add_argument("--output", required=True)
    capsule = sub.add_parser("capsule")
    capsule.add_argument("--root", required=True)
    capsule.add_argument("--files", nargs="+", required=True)
    capsule.add_argument("--output", required=True)
    capsule.add_argument("--id", default="evidence")
    capsule.add_argument("--html")
    extract = sub.add_parser("extract")
    extract.add_argument("--notebook", required=True)
    extract.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "compact":
        report = compact_checkpoint(args.source, args.output)
    elif args.command == "capsule":
        report = create_capsule(args.root, args.files, args.output, capsule_id=args.id)
        if args.html:
            _new_file(args.html, capsule_html(args.output).encode())
    else:
        report = extract_capsules(args.notebook, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
