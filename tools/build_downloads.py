"""Build deterministic, standalone ZIP downloads from curated project source."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
             '.runtime', 'runtime', '.local', '.local_support', '.test-runtime',
             'run_output', '.ipynb_checkpoints', '.ruff_cache'}
EXTENSIONS = {'.py', '.md', '.txt', '.json', '.sql', '.csv', '.pdf', '.ipynb', '.toml', '.ini'}


def project_files(project):
    for path in sorted(project.rglob('*')):
        relative = path.relative_to(project)
        if not path.is_file() or any(part in SKIP_DIRS or part.startswith('.test-runtime') for part in relative.parts):
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.suffix in EXTENSIONS or path.name in {'.env.example', '.gitignore'}:
            yield path


def projects():
    return sorted(p for track in ('langchain', 'langgraph')
                  for p in (ROOT / track).iterdir() if p.is_dir() and (p / 'README.md').is_file())


def main():
    downloads = ROOT / 'downloads'
    downloads.mkdir(exist_ok=True)
    manifest = []
    for project in projects():
        name = project.relative_to(ROOT).as_posix().replace('/', '--') + '.zip'
        target = downloads / name
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in project_files(project):
                info = zipfile.ZipInfo(project.name + '/' + path.relative_to(project).as_posix(), (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        manifest.append({'project': project.relative_to(ROOT).as_posix(), 'archive': name,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'bytes': target.stat().st_size})
    (downloads / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Built {len(manifest)} project ZIPs ({sum(p["bytes"] for p in manifest):,} bytes).')


if __name__ == '__main__':
    main()
