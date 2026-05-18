#!/usr/bin/env python3
"""Download a YouTube subtitle track with yt-dlp and normalize it to Markdown."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse


TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d\d:\d\d:\d\d\.\d{3})\s+-->\s+(?P<end>\d\d:\d\d:\d\d\.\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    query_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_id:
        return query_id
    raise ValueError(f"Could not infer YouTube video id from URL: {url}")


def timestamp_to_seconds(timestamp: str) -> int:
    hours, minutes, seconds = timestamp.split(":")
    whole_seconds = seconds.split(".")[0]
    return int(hours) * 3600 + int(minutes) * 60 + int(whole_seconds)


def clean_caption_text(line: str) -> str:
    line = TAG_RE.sub("", line)
    line = html.unescape(line)
    return " ".join(line.split())


def parse_vtt(vtt_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    current_start: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_lines
        if current_start is None:
            return
        text = " ".join(filter(None, (clean_caption_text(line) for line in current_lines)))
        if text and (not entries or entries[-1][1] != text):
            entries.append((current_start, text))
        current_start = None
        current_lines = []

    for raw_line in vtt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        match = TIMESTAMP_RE.match(line)
        if match:
            flush()
            current_start = match.group("start")
            continue
        if line in {"WEBVTT"} or line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if current_start is not None:
            current_lines.append(line)

    flush()
    return entries


def write_markdown(
    *,
    output_path: Path,
    title: str,
    source_url: str,
    video_id: str,
    language: str,
    entries: list[tuple[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# {title}",
        "",
        f"- Source: {source_url}",
        f"- Video ID: `{video_id}`",
        f"- Transcript language: `{language}`",
        f"- Fetched: {fetched_at}",
        "",
        "## Transcript",
        "",
    ]
    for start, text in entries:
        seconds = timestamp_to_seconds(start)
        lines.append(f"- [{start}](https://www.youtube.com/watch?v={video_id}&t={seconds}s) {text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_yt_dlp(url: str, output_template: Path, language: str) -> None:
    command = [
        "yt-dlp",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        language,
        "--sub-format",
        "vtt",
        "--skip-download",
        "--output",
        str(output_template),
        url,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--title", default="YouTube Transcript", help="Markdown title")
    parser.add_argument("--language", default="en-US", help="Subtitle language code")
    parser.add_argument("--output-root", type=Path, default=Path("references") / "youtube")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only normalize an already downloaded VTT file",
    )
    args = parser.parse_args()

    video_id = video_id_from_url(args.url)
    video_dir = args.output_root / video_id
    output_template = video_dir / f"{video_id}.%(ext)s"
    vtt_path = video_dir / f"{video_id}.{args.language}.vtt"
    markdown_path = video_dir / f"transcript.{args.language}.md"

    video_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        run_yt_dlp(args.url, output_template, args.language)
    if not vtt_path.exists():
        raise FileNotFoundError(f"Expected subtitle file was not found: {vtt_path}")

    entries = parse_vtt(vtt_path)
    write_markdown(
        output_path=markdown_path,
        title=args.title,
        source_url=args.url,
        video_id=video_id,
        language=args.language,
        entries=entries,
    )
    print(f"Wrote {markdown_path} with {len(entries):,} transcript entries.")


if __name__ == "__main__":
    main()
