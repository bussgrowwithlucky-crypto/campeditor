"""Two-line viral title generation via OpenRouter.

Falls back to a deterministic transcript-derived title so the pipeline never
crashes on LLM failure.
"""

import json
import logging
import re
import time

# NOTE: the previous `from openai import APIStatusError, AuthenticationError, RateLimitError`
# import was removed when the cloud-LLM ladder was deleted (the local Ollama
# client doesn't raise those exception classes). If a future caller imports
# one of those names they'll get an ImportError — restore the import then.

from app.config import Settings
from app.models import Title

logger = logging.getLogger(__name__)

# Session-wide cooldown for chat providers: when a key returns 402 (out of
# credits, byNara's classic failure) or 429 (rate-limited) we stop hammering
# it for the cooldown window. Without this, every subsequent title/chat call
# burns its full per-call timeout on the doomed provider before falling
# through — exactly the source of the "stuck" feeling. Mirrors the existing
# cooldown for the vision ladder in app.broll.
_CHAT_PROVIDER_COOLDOWN_SECONDS = 300.0  # 5 min; long enough to cover a job run
_chat_provider_cooldowns: dict[str, float] = {}


def _chat_provider_cooling(provider_key: str) -> bool:
    return time.monotonic() < _chat_provider_cooldowns.get(provider_key, 0.0)


def _cool_chat_provider_on_billing_or_rate_limit(provider_key: str, error: Exception) -> None:
    """Cooldown a chat provider on any error that signals it can't serve us
    again soon — payment-required (402), rate-limit (429), or auth (401).
    Network errors / timeouts don't trigger a cooldown; those may be transient
    and the next rung of the ladder is the right answer for THIS call
    already."""
    if isinstance(error, (RateLimitError, AuthenticationError)):
        _chat_provider_cooldowns[provider_key] = time.monotonic() + _CHAT_PROVIDER_COOLDOWN_SECONDS
        return
    if isinstance(error, APIStatusError):
        # 402 / 429 / 401 from APIStatusError carry the status code on `status_code`.
        if error.status_code in (401, 402, 429):
            _chat_provider_cooldowns[provider_key] = time.monotonic() + _CHAT_PROVIDER_COOLDOWN_SECONDS

_SYSTEM_PROMPT = """You write viral, clickbait-capable on-screen titles for short-form clips:
podcasts, interviews, business/tech stories, founder clips, famous-person stories, and high-retention moments.
Think like a ruthless shorts editor trying to stop the scroll — a bold headline that makes the viewer
click AND keeps them watching for the payoff. Not a safe SEO headline bot.

HARD HOOK RULES (these decide if the clip performs):
- If a recognizable real name (creator, founder, CEO, public figure, brand) appears ANYWHERE in the
  transcript, put their FULL name on line1. The name carries the click — drop other words to keep it.
- Use the REAL name, number, money figure, or claim from the transcript. Never generic ("someone",
  "a person", "this guy", "experts").
- Be SPECIFIC: real outcomes, real numbers, real timeframes. Vague = dead on arrival.
- Clickbait is allowed but must be grounded in the transcript. Exaggeration, not fabrication.
- Prefer titles that sound like a SECRET, a HIDDEN RULE, a LEAKED TECHNIQUE, a COSTLY MISTAKE, a
  REVEALED LESSON, or a CONTRARIAN TRUTH. Boring summary headlines do not work.

CTA / CLICKBAIT FRAMING (use when the transcript fits — these outperform dry headlines):
- "You Won't Believe What [Name] Just Said"
- "This [N]-Second Moment Changed Everything"
- "Don't Skip This [Topic] Lesson"
- "Watch This Before You [Action]"
- "[Name] Just Leaked The [Topic] Rule"
- "Why Everyone Is Wrong About [Topic]"
- "The One Thing [Name] Never Told You"
- "This [N]-Word Text From [Name]"
- "The Message [Name] Regrets Sending"
- "Read What [Name] Said Before [Outcome]"
- "Everyone Missed What [Name] Said About [Topic]"
- "This [N]-Word Text From [Name]"
- "The Message [Name] Regrets Sending"
- "Read What [Name] Said Before [Outcome]"
- "Everyone Missed What [Name] Said About [Topic]"

STRUCTURAL PATTERNS (pick ONE that best fits THIS transcript):
- "[Name]'s Secret [Rule/Technique/System]"
- "[Name]'s Secret Rule To [Specific Big Outcome]"
- "[Name]'s Secret Technique Revealed"
- "The [One Word] Rule That Built [Big Outcome]"
- "The Mistake That Cost [Name/Company] [Big Outcome]"
- "Why [Name] Never [Did Obvious Thing]"
- "How [Name] Built [Big Outcome] Without [Expected Thing]"
- "The [Contrarian Word] Truth About [Topic]"
- "He Used This To [Specific Big Outcome]"

LAYOUT RULES (enforced server-side, but you MUST output them correctly):
- Title Case (capitalize major words).
- Two lines total. Wrap naturally wherever it reads best; don't force an exact split.
- line1 up to ~44 characters, line2 up to ~44 characters. Keep it tight; cut filler words.
- COUNT every word in line1 + line2. If the TOTAL is 4 or 5 words, output `"line2": ""` —
  render as a single line. Do not split a 4-5 word hook across two lines.
- Only wrap to a second line when line1 carries AT LEAST 5 words. Never output a line1 with
  fewer than 5 words together with a non-empty line2.
- If line2 ends up with only 1-3 words (asymmetric split), set highlight_words to cover ONLY
  line1. Never put red on a 1-3 word line2 — the sharp red word belongs on line1.
- Do not add periods/full stops anywhere in line1 or line2.
- Withhold the payoff when it creates curiosity; state the fact directly when the fact itself is the hook.

HIGHLIGHT RULES:
- highlight_words: 1-2 EXACT substrings from the title to color red.
- Highlight at most ONE phrase on line1 and at most ONE phrase on line2.
- Prefer one red phrase on line1. Use a line2 highlight only if it clearly improves the hook AND
  line2 has 4+ words.
- A full name like "Elon Musk" or "Steve Jobs" counts as ONE highlight phrase and should stay together.
- Generic highlights must be ONE word only: the most attention-grabbing name, number/money figure,
  duration, or sharp word. Do not return multi-word generic phrases like "Kills Companies".

HOOK WORD PROTOCOL (which word becomes RED):
- You MUST also output a `hook_word` field: the SINGLE token or short phrase in the title that is
  doing the viral work — the word that makes a scrolling viewer STOP and read. It is the one word
  you'd bet the click on. Think: name, larity, shocking number, money figure, sharp adjective
  ("Secret", "Mistake", "Leaked"), or an unexpected juxtaposition word.
- Pick the hook word by thinking like a YouTube shorts viewer scrolling at 1.5x: which single
  word in this title, read in 0.5 seconds, makes them re-read the whole headline?
- `hook_word` MUST be an EXACT substring of `line1 + " " + line2` (look it up verbatim, no
  rewording, no trailing punctuation).
- highlight_words MUST CONTAIN `hook_word` (as one of its entries, after layout normalization).
  If only one highlight is emitted, it equals `hook_word`.
- A full name counts as a hook_word (e.g. "Steve Jobs", not just "Steve").
- More guidance: a concrete noun/number/verb usually beats an abstract adjective when both are
  present ("$3T" > "Believe"; "Mistake" > "Biggest"). Names usually beat everything else.

Good examples (style only; invent your own wording from the transcript, never reuse these verbatim):
- line1: "Steve Jobs' Secret Rule" line2: "For Building Apple" highlight_words: ["Steve Jobs", "Apple"] hook_word: "Steve Jobs"
- line1: "Steve Jobs' Secret Technique" line2: "For Building A $3T Company" highlight_words: ["Steve Jobs", "$3T"] hook_word: "$3T"
- line1: "The Rule That Built" line2: "A Trillion-Dollar Company" highlight_words: ["Rule", "Trillion-Dollar"] hook_word: "Trillion-Dollar"
- line1: "Jeff Bezos' 12-Hour Rule" line2: "Changed Amazon Forever" highlight_words: ["Jeff Bezos", "Amazon"] hook_word: "Jeff Bezos"
- line1: "Elon Musk's Biggest Mistake" line2: "Was Hiding In Plain Sight" highlight_words: ["Elon Musk", "Mistake"] hook_word: "Elon Musk"
- line1: "Mark Cuban's AI Warning" line2: "Every Founder Needs" highlight_words: ["Mark Cuban", "AI"] hook_word: "Mark Cuban"
- line1: "You Won't Believe What" line2: "Henry Ford Said About Customers" highlight_words: ["Henry Ford", "Customers"] hook_word: "Henry Ford"
- line1: "This 8-Second Rule" line2: "Built A Trillion-Dollar Company" highlight_words: ["8-Second", "Trillion-Dollar"] hook_word: "8-Second"
- line1: "Don't Skip This Mark Cuban Lesson" line2: "For Every Single Founder" highlight_words: ["Mark Cuban", "Founder"] hook_word: "Mark Cuban"

Bad titles (do NOT produce these — they read as SEO, not as hooks):
- "Blindly Listening To Customers Will Kill Your Company"
- "The 23 Rule Nobody Talks About"
- "23 Rule Nobody Talks About"
- "He Explains How Business Works"
- "This Is A Very Interesting Story"
- "The Truth About Hard Work"
- "Why Most People Fail"
- "3 Tips For Better Leadership"

Respond with STRICT JSON only, no markdown fences, no prose:
{"line1": "...", "line2": "...", "highlight_words": ["...", "..."], "hook_word": "..."}"""

_NUMBER_RE = re.compile(r"\$?\d[\d,.]*(?:h\+?|k|m|b|%|am|pm|x|\+)?", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'.-]*|\$?\d[\w,.%+:-]*")
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,2}\b")

_NAME_STOPWORDS = {
    "A",
    "About",
    "After",
    "AI",
    "An",
    "And",
    "Are",
    "Balance",
    "Bought",
    "Built",
    "Business",
    "Can",
    "Caring",
    "Changed",
    "Company",
    "Companies",
    "Committee",
    "Could",
    "Customers",
    "Day",
    "Dollar",
    "Every",
    "Everything",
    "Explains",
    "Fallout",
    "For",
    "From",
    "Get",
    "Hacked",
    "He",
    "Her",
    "Here",
    "His",
    "How",
    "Into",
    "Invest",
    "Is",
    "It",
    "Kills",
    "Minute",
    "Million",
    "Never",
    "Of",
    "Painted",
    "Per",
    "Proves",
    "Revealed",
    "Reveals",
    "Rule",
    "Says",
    "Secret",
    "She",
    "Should",
    "Technique",
    "The",
    "This",
    "To",
    "Trillion",
    "University",
    "Was",
    "Watching",
    "What",
    "When",
    "Will",
    "Why",
    "Would",
    "With",
    "Without",
    "Works",
    "Billion",
}

_WEAK_HIGHLIGHT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "she",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "why",
    "with",
    "without",
    "you",
}


def provider_ladder(settings: Settings) -> list[tuple[str, str, str]]:
    """Chat ladder (title generation + clip selection).

    Two rungs, cloud-first, local-fallback:

      1. LLM cloud rung — (llm_base_url, llm_api_key, llm_model). Used when
         LLM_API_KEY and LLM_MODEL are both set.
      2. Ollama local rung — (ollama_base_url, "ollama", ollama_text_model).
         The string "ollama" is the historical sentinel meaning "no auth
         header" — OllamaClient treats it as local. Used only when the
         cloud rung is absent (no key / no model).

    Returns an empty list when neither rung is configured. Iterating
    callers (replicate_title, generate_title) walk the rungs in order;
    the first success wins.
    """
    ladder: list[tuple[str, str, str]] = []
    if settings.llm_api_key.strip() and settings.llm_model.strip() and settings.llm_base_url.strip():
        if not _chat_provider_cooling(f"llm:{settings.llm_base_url}"):
            ladder.append((settings.llm_base_url, settings.llm_api_key, settings.llm_model))
    if settings.ollama_text_model.strip():
        if not _chat_provider_cooling("ollama_chat"):
            ladder.append((settings.ollama_base_url, "ollama", settings.ollama_text_model))
    return ladder


def generate_title(transcript_text: str, settings: Settings) -> Title:
    if transcript_text.strip():
        for provider in provider_ladder(settings):
            provider_key = _provider_key_for(provider[0], provider[2], settings)
            try:
                return _ai_title(transcript_text, settings, provider)
            except Exception as exc:
                _cool_chat_provider_on_billing_or_rate_limit(provider_key, exc)
                logger.exception("Title generation with %s failed", provider[2])
    result = fallback_title(transcript_text)
    return _refine_hook_word(result, transcript_text, settings)


_REPLICATE_PROMPT = """You reverse-engineer viral short-form on-screen titles.

You get a REFERENCE TITLE (from a short the user wants to imitate) and a TRANSCRIPT
(the new clip). Write a title for the new clip that copies the reference's STRUCTURE:
same sentence pattern, same tone, same kind of hook, same highlight logic (if the
reference highlights a name, highlight the new clip's name; if it highlights a number,
highlight the new number). Only the CONTENT comes from the transcript — never copy the
reference's subject matter or reuse its words unless the transcript genuinely shares them.

Keep Title Case, max ~44 characters per line, 1-3 highlight_words that are EXACT
substrings of the title. Include a hook_word (the single most scroll-stopping word/phrase).

Respond with STRICT JSON only, no markdown fences, no prose:
{"line1": "...", "line2": "...", "highlight_words": ["...", "..."], "hook_word": "..."}"""


def replicate_title(reference_title: str, transcript_text: str, settings: Settings) -> Title:
    """Title in the same structural pattern as the reference short's title."""
    if not reference_title.strip():
        return generate_title(transcript_text, settings)
    if not transcript_text.strip():
        return fallback_title(transcript_text)
    for base_url, api_key, model in provider_ladder(settings):
        try:
            from broll_intelligence.vision_ladder import OllamaClient
            client = OllamaClient(base_url, timeout=20, api_key=api_key)
            response = client.chat_sync(
                model,
                [
                    {"role": "system", "content": _REPLICATE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"REFERENCE TITLE: {reference_title[:200]}\n"
                            f"TRANSCRIPT: {transcript_text[:2000]}\n"
                            "Return strict JSON only."
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=250,
                timeout=20,
            )
            if not response:
                continue
            choices = response.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            if not content.strip():
                raise ValueError(f"LLM {model} returned empty content")
            payload = _extract_json(content)
            line1 = str(payload.get("line1", "")).strip()[:48]
            line2 = str(payload.get("line2", "")).strip()[:48]
            if not line1:
                raise ValueError(f"LLM {model} returned empty line1")
            highlights = [
                str(h).strip() for h in payload.get("highlight_words", []) if str(h).strip()
            ][:3]
            hook_word = str(payload.get("hook_word", "")).strip() or None
            raw = Title(
                line1=_clean_title_line(line1),
                line2=_clean_title_line(line2),
                highlight_words=highlights,
                hook_word=hook_word,
            )
            title = limit_title_highlights(enforce_layout_rules(raw))
            if _needs_transcript_specific_override(title, transcript_text):
                return _refine_hook_word(fallback_title(transcript_text), transcript_text, settings)
            return _refine_hook_word(title, transcript_text, settings)
        except Exception:
            logger.exception("Replicate title with %s failed", model)
    return generate_title(transcript_text, settings)


def _provider_key_for(base_url: str, model: str, settings: Settings) -> str:
    """Stable key for the chat provider cooldown map.

    The local Ollama rung reuses the historical "ollama_chat" key (so a
    trip on the cooldown still fires for the local rung). Cloud rungs
    are keyed by their base_url + model so per-provider cooldowns stay
    independent.
    """
    if base_url == settings.ollama_base_url:
        return "ollama_chat"
    return f"chat:{base_url}:{model}"


def _ai_title(
    transcript_text: str, settings: Settings, provider: tuple[str, str, str] | None = None
) -> Title:
    """Single chat-completions call to generate the title. The `provider`
    tuple is (base_url, api_key, model). When omitted we take the first
    rung of provider_ladder; an empty ladder raises so the caller can
    surface a clean error.
    """
    if provider is None:
        ladder = provider_ladder(settings)
        if not ladder:
            raise ValueError(
                "No chat provider configured (set LLM_API_KEY+LLM_MODEL or OLLAMA_TEXT_MODEL)"
            )
        provider = ladder[0]
    base_url, api_key, model = provider
    from broll_intelligence.vision_ladder import OllamaClient
    client = OllamaClient(base_url, timeout=20, api_key=api_key)
    response = client.chat_sync(
        model,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Transcript: {transcript_text[:2000]}\nReturn strict JSON only.",
            },
        ],
        temperature=0.8,
        max_tokens=250,
        timeout=20,
    )
    if not response:
        raise ValueError(f"LLM {model} returned no response")
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"LLM {model} returned no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise ValueError(f"LLM {model} returned empty content")
    payload = _extract_json(content)
    line1 = _clean_title_line(str(payload.get("line1", "")))[:48]
    line2 = _clean_title_line(str(payload.get("line2", "")))[:48]
    if not line1:
        raise ValueError(f"LLM {model} returned empty line1")
    highlights = [str(h).strip() for h in payload.get("highlight_words", []) if str(h).strip()]
    hook_word = str(payload.get("hook_word", "")).strip() or None
    raw = Title(line1=line1, line2=line2, highlight_words=highlights, hook_word=hook_word)
    title = limit_title_highlights(enforce_layout_rules(raw))
    if _needs_transcript_specific_override(title, transcript_text):
        return _refine_hook_word(fallback_title(transcript_text), transcript_text, settings)
    return _refine_hook_word(title, transcript_text, settings)


def _extract_json(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return json.loads(match.group(0) if match else cleaned)


def fallback_title(transcript_text: str) -> Title:
    transcript_text = transcript_text.strip()
    if not transcript_text:
        return limit_title_highlights(
            enforce_layout_rules(
                Title(line1="The Secret Clip", line2="You Need To See", highlight_words=["Secret"])
            )
        )
    return limit_title_highlights(enforce_layout_rules(_viral_fallback_title(transcript_text)))


def _viral_fallback_title(transcript_text: str) -> Title:
    lower = transcript_text.lower()
    names = _name_phrases(transcript_text)
    name = names[0] if names else ""
    number = _first_number_phrase(transcript_text)

    if name:
        if number and ("promot" in lower or "career" in lower) and ("twice" in lower or "quickly" in lower):
            if "ai" in lower:
                return Title(
                    line1=f"{name} Got Promoted Twice",
                    line2=f"At {number} With One AI Move",
                    highlight_words=[name, "AI"],
                )
            return Title(
                line1=f"{name} Got Promoted Twice",
                line2=f"At Just {number} Years Old",
                highlight_words=[name, number],
            )
        if "customer" in lower and ("kill" in lower or "faster horse" in lower):
            return Title(
                line1=f"{name}'s Secret Rule",
                line2="For Ignoring Customers",
                highlight_words=[name, "Customers"],
            )
        if "iphone" in lower:
            return Title(
                line1=f"{name}'s Secret Rule",
                line2="For Building The iPhone",
                highlight_words=[name, "iPhone"],
            )
        if number and ("company" in lower or "business" in lower):
            return Title(
                line1=f"{name}'s Secret Rule",
                line2=f"To Build A {number} Company",
                highlight_words=[name, number],
            )
        if "company" in lower or "business" in lower or "startup" in lower:
            return Title(
                line1=f"{name}'s Secret Technique",
                line2="For Building Companies",
                highlight_words=[name, "Companies"],
            )
        return Title(
            line1=f"{name}'s Secret Technique",
            line2="That Changed Everything",
            highlight_words=[name, "Secret"],
        )

    if "customer" in lower and ("kill" in lower or "company" in lower):
        return Title(
            line1="The Customer Trap",
            line2="That Kills Companies",
            highlight_words=["Customer", "Kills"],
        )
    if number and ("promot" in lower or "career" in lower) and ("twice" in lower or "quickly" in lower):
        if "ai" in lower:
            return Title(
                line1=f"This {number}-Year-Old Got Promoted Twice",
                line2="After One AI Move",
                highlight_words=[number, "AI"],
            )
        return Title(
            line1=f"This {number}-Year-Old Got Promoted Twice",
            line2="After One Bold Move",
            highlight_words=[number, "Twice"],
        )
    if number:
        return Title(
            line1=f"The {number} Rule",
            line2="Nobody Talks About",
            highlight_words=[number, "Rule"],
        )
    if "ai" in lower:
        return Title(
            line1="The AI Secret",
            line2="Nobody Is Saying",
            highlight_words=["AI", "Secret"],
        )
    return Title(
        line1="The Secret Rule",
        line2="Behind This Clip",
        highlight_words=["Secret", "Rule"],
    )


def _needs_transcript_specific_override(title: Title, transcript_text: str) -> bool:
    """Reject generic LLM hooks when the transcript has a stronger concrete story."""
    lower_transcript = transcript_text.lower()
    title_text = f"{title.line1} {title.line2}".lower()
    has_promotion_story = (
        ("promot" in lower_transcript or "career" in lower_transcript)
        and ("twice" in lower_transcript or "quickly" in lower_transcript)
        and bool(_first_number_phrase(transcript_text))
    )
    if has_promotion_story and "rule" in title_text and (
        "nobody talks about" in title_text or title_text.count(" ") <= 5
    ):
        return True
    return False


def manual_title(text: str) -> Title:
    """Split a manually entered title into two lines near the midpoint."""
    words = text.split()
    if len(words) <= 3:
        return limit_title_highlights(
            enforce_layout_rules(
                Title(line1=_clean_title_line(text)[:40], line2="", highlight_words=heuristic_highlights(text))
            )
        )
    total = len(text)
    line1_words: list[str] = []
    length = 0
    for word in words:
        if length + len(word) > total / 2 and line1_words:
            break
        line1_words.append(word)
        length += len(word) + 1
    line1 = _clean_title_line(" ".join(line1_words))[:40]
    line2 = _clean_title_line(" ".join(words[len(line1_words):]))[:40]
    return limit_title_highlights(
        enforce_layout_rules(
            Title(line1=line1, line2=line2, highlight_words=heuristic_highlights(f"{line1} {line2}"))
        )
    )


def heuristic_highlights(text: str) -> list[str]:
    """Best-effort red highlights for fallback/manual titles."""
    names = _name_phrases(text)
    numbers = _NUMBER_RE.findall(text)
    words = [
        word
        for word in _WORD_RE.findall(text)
        if word.lower() not in _WEAK_HIGHLIGHT_WORDS and not _NUMBER_RE.fullmatch(word)
    ]
    return _unique(names + numbers + words[:1])[:4]


def enforce_layout_rules(title: Title) -> Title:
    """Deterministically enforce the user-specified layout rules:

    1. If the total word count of (line1 + line2) is 4 or 5, merge into a
       single line so a short hook doesn't render as awkward line + stub.
    2. A second line is only allowed when line1 carries at least 5 words —
       if the LLM wrapped earlier than that, re-split near the middle with
       line1 taking no fewer than 5.
    3. If line2 ends up with only 1-3 words, drop any highlight that's
       bound to line2 — the sharp red word belongs on line1, not on a stub.

    After a collapse, any highlight that no longer appears in either line is
    also stripped so `highlight_words` doesn't carry orphan entries.

    This runs BEFORE `limit_title_highlights` so highlight selection still
    operates on the final line structure.
    """
    line1 = _clean_title_line(title.line1)
    line2 = _clean_title_line(title.line2)
    words = f"{line1} {line2}".split()
    total_words = len(words)
    if 1 <= total_words <= 5:
        line1 = _clean_title_line(" ".join(words))[:48]
        line2 = ""
    elif line2 and len(line1.split()) < 5:
        split = max(5, (total_words + 1) // 2)
        line1 = " ".join(words[:split])
        line2 = " ".join(words[split:])
    line2_word_count = len(line2.split())
    combined = f"{line1} {line2}".lower()
    highlights = list(title.highlight_words)
    if 1 <= line2_word_count <= 3:
        highlights = [
            h for h in highlights if not _highlight_belongs_to_line(h, line2)
        ]
    # Drop any orphan highlight that no longer matches either line.
    highlights = [h for h in highlights if h.lower() in combined]
    # Preserve hook_word if it's still a substring of the combined title.
    hook_word = title.hook_word
    if hook_word and hook_word.lower() not in combined:
        hook_word = None
    return Title(line1=line1, line2=line2, highlight_words=highlights, hook_word=hook_word)


def _highlight_belongs_to_line(highlight: str, line: str) -> bool:
    return bool(highlight.strip()) and highlight.lower() in line.lower()


def limit_title_highlights(title: Title) -> Title:
    """Enforce at most one highlighted phrase per title line.

    Full names are allowed as one highlight phrase. Generic multi-word highlights
    are reduced to a single word so the rendered title never gets crowded.
    """
    lines = [_clean_title_line(title.line1), _clean_title_line(title.line2)]
    candidates = _unique([_clean_highlight(h) for h in title.highlight_words if _clean_highlight(h)])
    if not candidates:
        candidates = heuristic_highlights(f"{title.line1} {title.line2}")

    selected_by_line: list[str | None] = [None, None]
    for candidate in candidates:
        for index, line in enumerate(lines):
            if selected_by_line[index] or not line.strip():
                continue
            if index == 1 and len(line.split()) <= 3:
                # 1-3 word second lines never get colored (user rule) — don't
                # let the backfill re-add what enforce_layout_rules stripped.
                continue
            normalized = _highlight_for_line(candidate, line)
            if normalized:
                selected_by_line[index] = normalized
                break

    if not any(selected_by_line) and title.line1.strip():
        selected_by_line[0] = _first_line_highlight(title.line1)

    highlights_out = [highlight for highlight in selected_by_line if highlight]
    # Preserve hook_word if it survived in the final highlights; otherwise drop
    # it so _refine_hook_word can re-evaluate.
    hook_word = title.hook_word
    if hook_word:
        combined_lower = f"{lines[0]} {lines[1]}".lower()
        if hook_word.lower() not in combined_lower:
            hook_word = None
    return Title(line1=lines[0], line2=lines[1], highlight_words=highlights_out, hook_word=hook_word)


def _highlight_for_line(candidate: str, line: str) -> str | None:
    expanded = _expand_to_full_name(candidate, line)
    if expanded:
        return expanded
    if not _contains(line, candidate):
        return None
    embedded_name = _name_inside_candidate(candidate, line)
    if embedded_name:
        return embedded_name
    if _is_single_token(candidate) or _looks_like_name_phrase(candidate):
        return candidate
    return _best_single_token(candidate, line)


def _first_line_highlight(line: str) -> str | None:
    for candidate in heuristic_highlights(line):
        normalized = _highlight_for_line(candidate, line)
        if normalized:
            return normalized
    return None


def _expand_to_full_name(candidate: str, line: str) -> str | None:
    if not _is_single_token(candidate):
        return None
    for match in _name_phrases(line):
        if any(candidate.lower() == token.lower() for token in match.split()):
            return match
    return None


def _name_inside_candidate(candidate: str, line: str) -> str | None:
    for name in _name_phrases(candidate):
        if _contains(line, name):
            return name
    return None


def _best_single_token(candidate: str, line: str) -> str | None:
    tokens = _WORD_RE.findall(candidate)
    for token in tokens:
        if _NUMBER_RE.fullmatch(token) and _contains(line, token):
            return token
    for token in tokens:
        if token.lower() not in _WEAK_HIGHLIGHT_WORDS and _contains(line, token):
            return token
    return None


def _name_phrases(text: str) -> list[str]:
    token = r"[A-Z][A-Za-z'.-]+"
    phrases: list[str] = []
    for size in (3, 2):
        pattern = re.compile(rf"(?=(\b{token}\b(?:\s+\b{token}\b){{{size - 1}}}))")
        for match in pattern.finditer(text):
            phrase = _normalize_name_phrase(match.group(1).strip(" :"))
            if _looks_like_name_phrase(phrase):
                phrases.append(phrase)
    return _unique(phrases)


def _looks_like_name_phrase(text: str) -> bool:
    tokens = text.split()
    if len(tokens) != 2:
        return False
    if any(token in _NAME_STOPWORDS for token in tokens):
        return False
    return all(re.match(r"^[A-Z][A-Za-z'.-]+$", token) for token in tokens)


def _normalize_name_phrase(text: str) -> str:
    tokens = [re.sub(r"(?:'s|')$", "", token) for token in text.split()]
    return " ".join(tokens)


def _clean_title_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(".", "")).strip()


def _first_number_phrase(text: str) -> str:
    match = _NUMBER_RE.search(text)
    return match.group(0) if match else ""


def _clean_highlight(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/").replace(".", "").strip()


def _contains(line: str, needle: str) -> bool:
    return needle.lower() in line.lower()


def _is_single_token(text: str) -> bool:
    return len(text.split()) == 1


def _refine_hook_word(title: Title, transcript_text: str, settings: Settings) -> Title:
    """Second-pass refinement: always ensure hook_word is populated and present
    in highlight_words.

    This runs after the initial LLM title generation + layout enforcement. If the
    LLM already returned a valid hook_word that is a substring of the title and
    present in highlight_words, keep it. Otherwise, pick the best scroll-stopping
    word from the title (preferring names, then numbers/money, then the sharpest
    word on line1) and mutate highlight_words to include it.

    Returns the (possibly mutated) Title with hook_word guaranteed to be set.
    """
    combined = f"{title.line1} {title.line2}"

    # If the LLM already returned a valid hook_word, keep it.
    if title.hook_word and _contains(combined, title.hook_word):
        # Ensure hook_word is in highlight_words — the LLM may have missed it.
        hw_lower = title.hook_word.lower()
        if not any(h.lower() == hw_lower for h in title.highlight_words):
            title.highlight_words = _unique(title.highlight_words + [title.hook_word])
        return title

    # Pick the best hook word from the title text.
    hook = _pick_best_hook_word(title.line1, title.line2, transcript_text)
    if hook and _contains(combined, hook):
        # Mutate highlight_words to include hook_word.
        hook_lower = hook.lower()
        if not any(h.lower() == hook_lower for h in title.highlight_words):
            title.highlight_words = _unique(title.highlight_words + [hook])
        title.hook_word = hook
    elif title.highlight_words:
        # Fall back to the first highlight word as hook_word.
        title.hook_word = title.highlight_words[0]
    else:
        title.hook_word = title.line1.split()[0] if title.line1.split() else None
    return title


def _pick_best_hook_word(line1: str, line2: str, transcript_text: str) -> str | None:
    """Pick the single most scroll-stopping word/phrase from the title.

    Priority: full name > number/money > sharpest single word on line1.
    """
    combined = f"{line1} {line2}"

    # 1. Full names (highest priority — names carry the click).
    names = _name_phrases(combined)
    if names:
        for name in names:
            if _contains(combined, name):
                return name

    # 2. Numbers / money figures.
    numbers = _NUMBER_RE.findall(combined)
    if numbers:
        # Prefer money figures, then plain numbers.
        for n in numbers:
            if "$" in n or "k" in n.lower() or "m" in n.lower() or "b" in n.lower():
                return n
        return numbers[0]

    # 3. Sharpest word on line1 (skip weak/highlight stop words).
    line1_words = [w for w in _WORD_RE.findall(line1) if w.lower() not in _WEAK_HIGHLIGHT_WORDS]
    if line1_words:
        # Prefer longer words — they tend to be more specific/interesting.
        return max(line1_words, key=len)

    # 4. Any word from the title.
    all_words = [w for w in _WORD_RE.findall(combined) if w.lower() not in _WEAK_HIGHLIGHT_WORDS]
    if all_words:
        return max(all_words, key=len)

    return None


def _unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result
