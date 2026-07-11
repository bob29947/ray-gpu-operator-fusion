#!/usr/bin/env python3
"""Overlay this branch's committed Ray Data Python delta onto a stock wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


def _git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=text
    )


def _digest(data: bytes) -> tuple[str, str]:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return f"sha256={encoded.rstrip(b'=').decode()}", str(len(data))


def _new_info(name: str, epoch: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, time.gmtime(max(epoch, 315532800))[:6])
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _committed_overlay(
    repo: Path, base_commit: str, head_commit: str
) -> tuple[dict[str, bytes], set[str]]:
    changed = _git(
        repo,
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        f"{base_commit}..{head_commit}",
        "--",
        "python/ray/data",
        text=False,
    ).split(b"\0")
    overlays: dict[str, bytes] = {}
    removals: set[str] = set()
    if changed and changed[-1] == b"":
        changed.pop()
    if len(changed) % 2:
        raise RuntimeError("unexpected git --name-status output")
    for index in range(0, len(changed), 2):
        status_code = changed[index].decode()
        source_name = changed[index + 1].decode()
        if source_name.startswith("python/ray/data/tests/") or source_name == (
            "python/ray/data/BUILD.bazel"
        ):
            continue
        if not source_name.endswith(".py"):
            raise RuntimeError(
                "Python-overlay build cannot package changed production file "
                f"{source_name!r}"
            )
        wheel_name = source_name.removeprefix("python/")
        status_kind = status_code[0]
        if status_kind == "D":
            removals.add(wheel_name)
        elif status_kind in {"A", "M"}:
            overlays[wheel_name] = _git(
                repo, "show", f"{head_commit}:{source_name}", text=False
            )
        else:
            raise RuntimeError(
                f"unsupported git status {status_code!r} for {source_name}"
            )
    if not overlays:
        raise RuntimeError("the modified commit has no Ray Data Python overlay")
    return overlays, removals


def _rewrite_version(original: bytes, head_commit: str) -> bytes:
    text = original.decode()
    rewritten, count = re.subn(
        r'^commit = "[0-9a-f]+"$',
        f'commit = "{head_commit}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("stock wheel has an unexpected ray/_version.py")
    return rewritten.encode()


def build(args: argparse.Namespace) -> Path:
    repo = args.repo.resolve()
    stock_wheel = args.stock_wheel.resolve()
    output_dir = args.output_dir.resolve()
    head_commit = _git(repo, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head_commit):
        raise RuntimeError(f"invalid HEAD commit: {head_commit!r}")
    subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", args.base_commit, head_commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "python/ray/data"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet", "HEAD", "--", "python/ray/data"],
        check=True,
    )
    overlays, removals = _committed_overlay(repo, args.base_commit, head_commit)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", _git(repo, "show", "-s", "--format=%ct", head_commit).strip()))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / stock_wheel.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(stock_wheel) as source:
            names = set(source.namelist())
            version_name = "ray/_version.py"
            if version_name not in names:
                raise RuntimeError("stock wheel does not contain ray/_version.py")
            overlays[version_name] = _rewrite_version(source.read(version_name), head_commit)
            record_names = [name for name in names if name.endswith(".dist-info/RECORD") and "/" not in name.removesuffix("/RECORD")]
            if len(record_names) != 1:
                raise RuntimeError("stock wheel must contain exactly one top-level RECORD")
            record_name = record_names[0]
            original_rows = {
                row[0]: (row[1], row[2])
                for row in csv.reader(io.StringIO(source.read(record_name).decode()))
            }
            records = {
                name: fields
                for name, fields in original_rows.items()
                if name not in removals and name != record_name
            }
            for name, data in overlays.items():
                records[name] = _digest(data)
            record_stream = io.StringIO(newline="")
            writer = csv.writer(record_stream, lineterminator="\n")
            for name in sorted(records):
                writer.writerow((name, *records[name]))
            writer.writerow((record_name, "", ""))
            record_data = record_stream.getvalue().encode()

            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as target:
                written: set[str] = set()
                for info in source.infolist():
                    name = info.filename
                    if name == record_name or name in removals:
                        continue
                    if name in overlays:
                        target.writestr(info, overlays[name])
                    else:
                        with source.open(info) as source_file, target.open(info, "w") as target_file:
                            shutil.copyfileobj(source_file, target_file, length=1 << 20)
                    written.add(name)
                for name in sorted(set(overlays) - written):
                    target.writestr(_new_info(name, epoch), overlays[name])
                target.writestr(_new_info(record_name, epoch), record_data)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"modified_commit={head_commit}")
    print(f"modified_wheel={output}")
    print(f"modified_wheel_sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stock-wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
