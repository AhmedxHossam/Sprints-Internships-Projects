"""Dependency-free checks for the public source, documentation, and downloads."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
from urllib.parse import unquote
import zipfile

from build_downloads import ROOT, SKIP_DIRS, project_files, projects

SECRET_PATTERNS = {
    'Google API key': re.compile(rb'AIza[0-9A-Za-z_-]{35}'),
    'GitHub token': re.compile(rb'(?:gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{40,})'),
    'OpenAI token': re.compile(rb'sk-(?:proj-|svcacct-)?[0-9A-Za-z_-]{40,}'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'Langfuse secret': re.compile(rb'sk-lf-[0-9a-f-]{20,}'),
    'AWS access key': re.compile(rb'AKIA[0-9A-Z]{16}'),
}


def source_files():
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and not any(part in SKIP_DIRS or part.startswith('.test-runtime') for part in path.relative_to(ROOT).parts):
            yield path


def main():
    errors = []
    count = 0
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append(f'{relative}: private environment file must not be published')
        if path.suffix == '.zip':
            continue  # archive contents are compared byte-for-byte below
        data = path.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                errors.append(f'{relative}: possible {name}; value withheld')
        if path.suffix == '.py':
            try:
                ast.parse(data.decode('utf-8-sig'), filename=relative)
            except SyntaxError as exc:
                errors.append(f'{relative}: Python syntax: {exc.msg}')
        elif path.suffix in {'.json', '.ipynb'}:
            try:
                obj = json.loads(data)
                if path.suffix == '.ipynb':
                    assert obj['nbformat'] == 4
                    for cell in obj['cells']:
                        if cell['cell_type'] == 'code':
                            assert not cell['outputs'] and cell['execution_count'] is None
                            code = ''.join(cell['source'])
                            # Notebook magics are valid notebook syntax, not Python AST syntax.
                            if not any(line.lstrip().startswith(('!', '%')) for line in code.splitlines()):
                                ast.parse(code)
            except (ValueError, KeyError, AssertionError, SyntaxError) as exc:
                errors.append(f'{relative}: invalid JSON/notebook ({type(exc).__name__})')
        elif path.suffix == '.md':
            for link in re.findall(r'\]\(([^\n)]+)\)', data.decode('utf-8-sig')):
                link = link.strip('<>').split(' "')[0]
                if not link or link.startswith(('#', 'https://', 'http://', 'mailto:')):
                    continue
                target = unquote(link.split('#', 1)[0])
                if not (path.parent / target).exists():
                    errors.append(f'{relative}: broken local link: {target}')
        count += 1
    manifest_path = ROOT / 'downloads/manifest.json'
    if not manifest_path.is_file():
        errors.append('Missing downloads manifest. Run python tools/build_downloads.py.')
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected_projects = {p.relative_to(ROOT).as_posix() for p in projects()}
        if {p['project'] for p in manifest} != expected_projects:
            errors.append('Download manifest does not cover all projects.')
        for entry in manifest:
            archive_path = ROOT / 'downloads' / entry['archive']
            if not archive_path.is_file():
                errors.append(f'Missing archive: {entry["archive"]}')
                continue
            if hashlib.sha256(archive_path.read_bytes()).hexdigest() != entry['sha256']:
                errors.append(f'Checksum mismatch: {entry["archive"]}')
            project = ROOT / entry['project']
            expected = {project.name + '/' + p.relative_to(project).as_posix(): p.read_bytes()
                        for p in project_files(project)}
            with zipfile.ZipFile(archive_path) as archive:
                actual = set(archive.namelist())
                if actual != set(expected) or any(archive.read(name) != data for name, data in expected.items() if name in actual):
                    errors.append(f'Stale download: {entry["archive"]}; rebuild downloads.')
                if archive.testzip() is not None:
                    errors.append(f'Corrupt archive: {entry["archive"]}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'PASS: {count} source/document files; {len(projects())} standalone project archives; syntax, links, notebook hygiene, checksums, and credential patterns.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
