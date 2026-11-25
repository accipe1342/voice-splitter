# Changelog

## [0.1.0] — 2026-11-12

First public release.

**Highlights**
- Transcribe any audio or video file and label every line with who said it.
- Review and correct the transcript and speakers, then export clean clips per speaker with their text.
- Guided first-run setup: checks your PC, connects the speaker model and downloads the transcription model that suits your graphics card.
- One double-click install on Windows: finds or installs Python, picks the GPU or CPU build of PyTorch, and brings its own ffmpeg.

### Added
- Review screen with a zoomable waveform, drag-to-mark ranges and per-voice colors.
- Lines, Script and Transcript views; edit the transcript like a document.
- Re-split a chosen range without losing the rest of your edits, and switch back to the previous split in one click.
- A queue: split many files one at a time with progress, time left, cancel, reorder and retry; the PC stays usable while it works.
- Status pills on every file (queued, splitting, ready to review, in review, ready to export, exported, failed).
- Your own labels on files, with a label filter.
- Download just the transcript (.txt or .srt) from the review screen, without exporting clips.
- Delete from the library: remove a file's split, or move the file itself to the Recycle Bin.
- Filters for lines to check, left out or flagged, and library filters by review stage.
- Voiceprint check that flags lines that sound like a different speaker.
- Export the full transcript as text and SRT subtitles, labelled by speaker.
- Export to per-speaker folders with 2–12 second clips and a GPT-SoVITS-format `dataset.list` of each clip's text.
