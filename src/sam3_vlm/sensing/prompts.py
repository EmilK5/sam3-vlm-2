"""Deterministic English prompt morphology without rewriting target identity."""

from functools import lru_cache
import re


# singular_noun assumes plural input. Protect already singular words that its
# suffix rules otherwise damage, and nouns whose plural denotes one object.
_KEEP = frozenset({"lens", "canvas", "news", "series", "species", "means", "deer", "fish", "sheep",
                   "scissors", "pliers", "tweezers", "pants", "trousers", "shorts", "jeans",
                   "clothes", "eyeglasses", "sunglasses", "binoculars", "headphones"})


@lru_cache(maxsize=1)
def _engine():
    import inflect
    return inflect.engine()


@lru_cache(maxsize=2048)
def singularize_prompt(prompt: str) -> str:
    """Singularize noun forms, including compound modifiers, preserving wording.

    This is morphology, not category inference: 'donuts tray' becomes 'donut
    tray', never silently 'donut'. Unknown/uninflected words remain unchanged.
    """
    if not isinstance(prompt, str):
        raise ValueError("SAM3 prompt must be a string")
    words = []
    for token in prompt.split():
        lower = token.lower()
        if (not re.fullmatch(r"[A-Za-z]+", token) or lower in _KEEP
                or (lower == "glasses" and any(w.lower() in {"reading", "prescription", "safety", "protective"} for w in words))
                or lower.endswith(("ss", "us", "is"))):
            words.append(token)
            continue
        singular = _engine().singular_noun(lower)
        if not singular:
            words.append(token)
        elif token.isupper():
            words.append(singular.upper())
        elif token.istitle():
            words.append(singular.title())
        else:
            words.append(singular)
    return " ".join(words)


def sensor_prompt(prompt: str, config) -> str:
    return singularize_prompt(prompt) if config.sam3.singularize_prompts else prompt
