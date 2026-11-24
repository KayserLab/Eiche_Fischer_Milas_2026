"""Extract release ZIPs safely into a fresh short directory; no registry changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def native_path(path):
    text = os.path.abspath(os.fspath(path))
    if os.name != 'nt' or text.startswith('\\\\?\\'):
        return text
    return '\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text


def inspect_targets(archives, destination):
    seen, total, longest, count = set(), 0, 0, 0
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for item in handle.infolist():
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                        or ':' in item.filename or not path.parts
                        or any(p.endswith((' ', '.')) for p in path.parts)
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError(f'Unsafe archive path: {item.filename}')
                target = destination.joinpath(*path.parts)
                key = os.path.normcase(str(target))
                if key in seen:
                    raise ValueError(f'Duplicate target: {target}')
                seen.add(key)
                longest = max(longest, len(str(target)))
                if len(str(target)) > 240:
                    raise ValueError(f'Path has {len(str(target))} characters: {target}. '
                                     'Choose a shorter fresh destination, e.g. C:\\EF26.')
                total += item.file_size
                count += 1
    return count, total, longest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-zip', type=Path)
    parser.add_argument('--data-zip', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--checksums', type=Path)
    args = parser.parse_args()
    destination = args.destination.expanduser().absolute()
    if os.path.exists(native_path(destination)):
        raise FileExistsError(f'Preserving existing attempt: {destination}; choose a fresh directory.')
    archives = [p.resolve() for p in (args.code_zip, args.data_zip) if p is not None]
    checksums = args.checksums or args.data_zip.parent / 'CHECKSUMS.sha256'
    expected = dict((name.lstrip('*'), digest.lower()) for digest, name in
                    (line.split(None, 1) for line in checksums.read_text(encoding='ascii').splitlines() if line.strip()))
    identities = {}
    for archive in archives:
        print(f'Checking SHA256: {archive.name}', flush=True)
        with archive.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected.get(archive.name):
            raise ValueError(f'Missing or incorrect supplied checksum: {archive}')
        identities[archive.name] = actual
    count, total, longest = inspect_targets(archives, destination)
    existing = destination.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < total + 1024**3:
        raise OSError('Insufficient disk space for extraction plus 1 GiB reserve')
    print(f'Preflight passed: {count} entries; longest target {longest} characters.', flush=True)
    os.makedirs(native_path(destination), exist_ok=False)
    record = dict(status='extracting', destination=str(destination), archives=identities,
                  entries=count, longest_target_characters=longest, uncompressed_bytes=total, extracted_files=0)
    try:
        for archive in archives:
            with zipfile.ZipFile(archive) as handle:
                for item in handle.infolist():
                    target = destination.joinpath(*PurePosixPath(item.filename).parts)
                    if item.is_dir():
                        os.makedirs(native_path(target), exist_ok=True)
                        continue
                    os.makedirs(native_path(target.parent), exist_ok=True)
                    with handle.open(item) as source, open(native_path(target), 'xb') as output:
                        shutil.copyfileobj(source, output, 1024**2)
                    record['extracted_files'] += 1
                    if record['extracted_files'] % 2000 == 0:
                        print(f"Extracted {record['extracted_files']} files", flush=True)
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with open(native_path(destination / 'extraction_record.json'), 'w', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
    print(f'Extraction complete: {destination}', flush=True)


if __name__ == '__main__':
    main()
