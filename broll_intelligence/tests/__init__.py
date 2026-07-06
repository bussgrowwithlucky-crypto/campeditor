"""broll_intelligence test suite — offline only.

The vision ladder is monkeypatched in every test (no API calls). ffmpeg and
ffprobe are NOT invoked — sample video files are tiny generated mp4s that
the indexer treats like real ones for mtime/size purposes, but the
extractor is also monkeypatched in test_library_indexer.py so the indexer
never actually opens them.
"""