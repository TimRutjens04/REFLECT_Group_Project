from __future__ import annotations
import re
from dataclasses import dataclass

from data_loader.task_loader import Task


# Recurring vocabulary across RoboFail trajectories. Maps colloquial / dataset
# wording to phrasing that Grounding DINO recognizes more reliably.
# Extend this as you encounter new objects during evaluation.
SYNONYMS: dict[str, list[str]] = {
    "mug": ["mug", "cup"],
    "cup": ["cup", "mug"],
    "bowl": ["bowl"],
    "plate": ["plate", "dish"],
    "bottle": ["bottle"],
    "can": ["can", "tin can"],
    "block": ["block", "cube"],
    "cube": ["cube", "block"],
    "drawer": ["drawer"],
    "shelf": ["shelf"],
    "table": ["table", "tabletop"],
    "robot": ["robot arm", "gripper"],
    "gripper": ["gripper", "robot arm"],
    "banana": ["banana"],
    "apple": ["apple"],
}

# Words to drop when extracting object nouns from free-text task descriptions.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "into",
    "onto",
    "from",
    "on",
    "in",
    "of",
    "with",
    "at",
    "by",
    "for",
    "up",
    "down",
    "is",
    "be",
    "put",
    "place",
    "pick",
    "move",
    "push",
    "pull",
    "grasp",
    "lift",
    "drop",
    "open",
    "close",
}


@dataclass
class PromptPlan:
    primary: str  # multi-object prompt, e.g. "red cup . blue plate ."
    singles: list[
        str
    ]  # per-object fallback prompts, e.g. ["red cup .", "blue plate ."]
    objects: list[str]  # canonical labels in the order they appear in primary


class PromptStrategy:
    """Builds Grounding DINO prompts from a RoboFail task description.

    Strategy:
      1. Take `object_list` from the task JSON as the canonical object set.
      2. Normalize each entry (lowercase, strip articles, collapse whitespace).
      3. Expand with synonyms only when needed (kept simple by default).
      4. Build a single multi-object prompt + per-object singles for fallback.
    """

    def __init__(self, expand_synonyms: bool = False):
        self.expand_synonyms = expand_synonyms

    def from_task(self, task: Task) -> PromptPlan:
        objects = self._extract_objects(task.object_list)
        if not objects:
            return PromptPlan(
                primary="object .", singles=["object ."], objects=["object"]
            )
        primary = self._format_multi(objects)
        singles = [self._format_single(o) for o in objects]
        return PromptPlan(primary=primary, singles=singles, objects=objects)

    def broaden(self, plan: PromptPlan) -> PromptPlan:
        """Recovery hook: expand each object with its synonym group."""
        expanded: list[str] = []
        seen: set[str] = set()
        for obj in plan.objects:
            for variant in SYNONYMS.get(self._head_noun(obj), [obj]):
                if variant not in seen:
                    seen.add(variant)
                    expanded.append(variant)
        return PromptPlan(
            primary=self._format_multi(expanded),
            singles=[self._format_single(o) for o in expanded],
            objects=expanded,
        )

    def _extract_objects(self, object_list: list[str]) -> list[str]:
        objects: list[str] = []
        for entry in object_list:
            cleaned = self._normalize(entry)
            if cleaned and cleaned not in objects:
                objects.append(cleaned)
        if self.expand_synonyms:
            objects = self._with_synonyms(objects)
        return objects

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [t for t in text.split() if t and t not in _STOPWORDS]
        return " ".join(tokens)

    @staticmethod
    def _head_noun(phrase: str) -> str:
        return phrase.split()[-1] if phrase else phrase

    def _with_synonyms(self, objects: list[str]) -> list[str]:
        out: list[str] = []
        for obj in objects:
            head = self._head_noun(obj)
            for variant in SYNONYMS.get(head, [obj]):
                if variant not in out:
                    out.append(variant)
        return out

    @staticmethod
    def _format_multi(objects: list[str]) -> str:
        # Grounding DINO requires lowercase + trailing dot, one per class.
        return " . ".join(objects) + " ."

    @staticmethod
    def _format_single(obj: str) -> str:
        return f"{obj} ."
