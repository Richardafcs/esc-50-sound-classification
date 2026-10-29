"""Resolve archived project-relative paths after the documented directory migration."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def project_path(value):
    path=Path(value)
    if path.is_absolute():
        try:value=str(path.relative_to(ROOT))
        except ValueError:return path
    else:value=str(path)
    direct=ROOT/value
    if direct.exists():return direct
    record=ROOT/'docs/project_layout_migration.json'
    if not record.exists():record=ROOT/'archive/reorganization_20261029/path_map.json'
    mapping=json.loads(record.read_text())['mapping'] if record.exists() else {}
    for old,new in sorted(mapping.items(),key=lambda item:len(item[0]),reverse=True):
        if value==old or value.startswith(old+'/'):
            return ROOT/(new+value[len(old):])
    return direct
