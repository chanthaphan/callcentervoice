"""Headless batch runner — process a folder of recordings without the web UI.

Usage:
    python -m app.batch [folder] [--workers N] [--reprocess] [--reanalyze] [--poll SECONDS]

Reads configuration from .env (same as the web app). Recurses into subfolders,
skips already-completed files (unless --reprocess/--reanalyze), and is safe to
re-run — interrupted batches resume where they left off.
"""
import argparse
import logging
import time
from pathlib import Path

from app.agent import PostCallAgent
from app.audio import TranscriptionService
from app.config import get_settings
from app.language import AzureLanguageService
from app.processor import BatchProcessor
from app.storage import CallStore

log = logging.getLogger("batch")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.batch", description=__doc__)
    parser.add_argument("folder", nargs="?", help="Folder to scan recursively (default: WATCH_FOLDER).")
    parser.add_argument("--workers", type=int, help="Parallel workers (overrides MAX_PARALLEL_FILES).")
    parser.add_argument("--reprocess", action="store_true", help="Re-transcribe and re-analyze even if complete.")
    parser.add_argument("--reanalyze", action="store_true", help="Re-run analysis on the existing transcript.")
    parser.add_argument("--poll", type=float, default=5.0, help="Progress-report interval in seconds.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    if args.workers:
        settings.max_parallel_files = args.workers
    folder = Path(args.folder) if args.folder else settings.absolute_watch_folder()

    store = CallStore(settings.absolute_data_dir())
    processor = BatchProcessor(
        store=store,
        transcription=TranscriptionService(settings),
        agent=PostCallAgent(settings),
        language_service=AzureLanguageService(settings),
        max_workers=settings.max_parallel_files,
    )

    log.info(
        "Scanning %s  (transcribe=%s, llm=%s, workers=%d)",
        folder, settings.transcribe_provider, settings.llm_provider, settings.max_parallel_files,
    )
    records = processor.process_folder(folder, force=args.reprocess, reanalyze=args.reanalyze)
    total = len(records)
    log.info("Enqueued %d file(s).", total)
    if total == 0:
        log.info("Nothing to do.")
        return

    # Progress loop. Counts are store-wide (cheap, indexed); the final tally below
    # is exact for this batch.
    try:
        while True:
            remaining = store.count("queued") + store.count("processing")
            log.info(
                "progress — complete=%d failed=%d remaining=%d",
                store.count("complete"), store.count("failed"), remaining,
            )
            if remaining == 0:
                break
            time.sleep(args.poll)
        processor.executor.shutdown(wait=True)
    except KeyboardInterrupt:
        log.warning("Interrupted — cancelling queued files, finishing in-flight; re-run to resume.")
        # cancel_futures drops not-yet-started tasks so we exit promptly instead of
        # draining the whole queue (which could be thousands of files).
        processor.executor.shutdown(wait=True, cancel_futures=True)

    done = failed = 0
    failures: list[tuple[str, str | None]] = []
    for stub in records:
        rec = store.get(stub.id)
        if not rec:
            continue
        if rec.status.value == "complete":
            done += 1
        elif rec.status.value == "failed":
            failed += 1
            failures.append((rec.file_name, rec.error))
    log.info("Done. %d complete, %d failed, of %d enqueued.", done, failed, total)
    for name, err in failures:
        log.warning("FAILED %s: %s", name, err)


if __name__ == "__main__":
    main()
