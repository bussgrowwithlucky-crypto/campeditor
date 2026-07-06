"""broll_intelligence — multi-dimensional B-roll selection system.

A sibling package to ``app/broll.py``. NOT integrated into the running
campeditor yet — kept isolated for offline testing. See ``CONTRACT.md``
for the public schema and ``README.md`` for usage.

Public surface at a glance::

    from broll_intelligence import (
        FeatureVector,                 # the schema
        extract_from_video,            # vision + CV extraction
        build_library_index,           # incremental library index
        get_settings,                  # pydantic-settings
    )
    from broll_intelligence.pipeline import select_broll, BrollPack, BrollItem
    from broll_intelligence.matcher import rank_candidates, score_clip
    from broll_intelligence.search import search_broll, BrollCandidate
    from broll_intelligence.baseline import local_score, rank_library_legacy

Run from the project root::

    python -m broll_intelligence.demo    --reference <video> --top-k 5
    python -m broll_intelligence.compare --reference <video> --top-k 5
"""

__version__ = "0.2.0"

# Public re-exports for ergonomics; keep imports here lazy-safe.
from .feature_vector import (  # noqa: F401  (re-export)
    FeatureVector,
    empty_feature_vector,
    feature_vector_to_dict,
    feature_vector_from_dict,
)
from .vibe_extractor import extract_from_video  # noqa: F401  (re-export)
from .library_indexer import build_library_index  # noqa: F401  (re-export)
from .config import Settings, get_settings  # noqa: F401  (re-export)