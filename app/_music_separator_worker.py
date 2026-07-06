"""Standalone subprocess entry point for MDX-Net instrumental separation.

Run in its own OS process (not in-thread inside the server) so ONNX
Runtime's CPU arena memory is fully returned to the OS on exit. In-process
separation was observed to fail with
``BFCArena::AllocateRawInternal Failed to allocate memory`` on back-to-back
calls within the same long-lived server process, even when calls were
serialized with a lock - the arena/session memory from a prior call wasn't
reliably released before the next one started. A subprocess guarantees a
clean memory slate for every separation.

Usage: python -m app._music_separator_worker <wav_path> <model_file_dir> <output_dir> <model_filename>
Prints the resulting instrumental filename (relative to output_dir) to
stdout and exits 0 on success; exits non-zero with a message on stderr
otherwise.
"""

import logging
import sys


def main() -> int:
    if len(sys.argv) != 5:
        print("expected 4 arguments: wav_path model_file_dir output_dir model_filename", file=sys.stderr)
        return 2
    wav_path, model_file_dir, output_dir, model_filename = sys.argv[1:5]

    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=output_dir,
        model_file_dir=model_file_dir,
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=model_filename)
    outputs = separator.separate(wav_path)
    instrumental = next((name for name in outputs if "instrumental" in name.lower()), None)
    if instrumental is None:
        print("no instrumental output produced", file=sys.stderr)
        return 1
    print(instrumental)
    return 0


if __name__ == "__main__":
    sys.exit(main())
