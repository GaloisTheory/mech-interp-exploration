"""BBQ (Bias Benchmark for QA) Dataset Loader.

Loads data from walledai/BBQ on HuggingFace.
Filters for ambiguous contexts where the correct answer is "Can't be determined".
"""

from dataclasses import dataclass
from typing import List, Optional
from datasets import load_dataset


# Mapping from user-friendly category names to dataset split names
CATEGORY_TO_SPLIT = {
    "age": "age",
    "disability": "disabilityStatus",
    "gender": "genderIdentity",
    "nationality": "nationality",
    "appearance": "physicalAppearance",
    "race": "raceEthnicity",
    "race_ses": "raceXSes",
    "race_gender": "raceXGender",
    "religion": "religion",
    "ses": "ses",
    "sexual_orientation": "sexualOrientation",
}

# All available categories
ALL_CATEGORIES = list(CATEGORY_TO_SPLIT.keys())


# Patterns that indicate an "unknown/can't determine" answer
UNKNOWN_PATTERNS = [
    "can't be determined",
    "cannot be determined",
    "can't answer",
    "cannot answer",
    "not enough info",
    "unknown",
    "undetermined",
    "not answerable",
    "can't tell",
    "cannot tell",
]


def _is_unknown_answer(answer_text: str) -> bool:
    """Check if an answer text indicates uncertainty/can't determine."""
    answer_lower = answer_text.lower().strip()
    return any(pattern in answer_lower for pattern in UNKNOWN_PATTERNS)


@dataclass
class BBQItem:
    """A single BBQ question item."""
    context: str
    question: str
    choices: List[str]  # [option_a, option_b, option_c]
    correct_idx: int    # 0, 1, or 2
    category: str
    item_idx: int       # Index in original dataset for reproducibility
    
    @property
    def correct_answer(self) -> str:
        """The correct answer text."""
        return self.choices[self.correct_idx]
    
    @property
    def correct_letter(self) -> str:
        """The correct answer as a letter (A, B, or C)."""
        return ["A", "B", "C"][self.correct_idx]
    
    @property
    def is_ambiguous(self) -> bool:
        """True if this is an ambiguous context (correct answer indicates uncertainty)."""
        return _is_unknown_answer(self.correct_answer)
    
    def format_prompt(self, template: str) -> str:
        """Format this item using a prompt template."""
        return template.format(
            context=self.context,
            question=self.question,
            choice_a=self.choices[0],
            choice_b=self.choices[1],
            choice_c=self.choices[2],
        )


def load_bbq_items(
    categories: Optional[List[str]] = None,
    n_per_category: Optional[int] = None,
    ambiguous_only: bool = True,
    seed: int = 42,
) -> List[BBQItem]:
    """Load BBQ items from walledai/BBQ.
    
    Args:
        categories: List of category names to load. If None, loads from all categories.
                   Valid names: age, disability, gender, nationality, appearance,
                               race, race_ses, race_gender, religion, ses, sexual_orientation
        n_per_category: Maximum number of items per category. If None, loads all.
        ambiguous_only: If True, only load items where correct_idx == 2 
                       (ambiguous contexts where answer is "Can't determine")
        seed: Random seed for reproducible sampling.
    
    Returns:
        List of BBQItem objects.
    """
    if categories is None:
        categories = ["ses", "age", "race"]  # Default to a few common ones
    
    # Validate category names
    for cat in categories:
        if cat.lower() not in CATEGORY_TO_SPLIT:
            valid = ", ".join(CATEGORY_TO_SPLIT.keys())
            raise ValueError(f"Unknown category '{cat}'. Valid categories: {valid}")
    
    # Load the dataset
    ds = load_dataset("walledai/BBQ")
    
    items = []
    for cat in categories:
        split_name = CATEGORY_TO_SPLIT[cat.lower()]
        split_data = ds[split_name]
        
        # Convert to list for filtering and sampling
        examples = list(split_data)
        
        # Filter for ambiguous if requested (correct answer is "Can't determine" variant)
        if ambiguous_only:
            examples = [ex for ex in examples if _is_unknown_answer(ex["choices"][ex["answer"]])]
        
        # Sample if n_per_category is specified
        if n_per_category is not None and len(examples) > n_per_category:
            # Deterministic sampling using seed
            import random
            rng = random.Random(seed)
            examples = rng.sample(examples, n_per_category)
        
        # Convert to BBQItem objects
        for idx, ex in enumerate(examples):
            items.append(BBQItem(
                context=ex["context"],
                question=ex["question"],
                choices=ex["choices"],
                correct_idx=ex["answer"],
                category=ex["category"],
                item_idx=idx,
            ))
    
    return items


def print_bbq_item(item: BBQItem, show_answer: bool = True):
    """Pretty print a BBQ item for inspection."""
    print(f"Category: {item.category}")
    print(f"Context: {item.context}")
    print(f"Question: {item.question}")
    print("Options:")
    for i, choice in enumerate(item.choices):
        letter = ["A", "B", "C"][i]
        marker = " *" if show_answer and i == item.correct_idx else ""
        print(f"  {letter}. {choice}{marker}")
    if show_answer:
        print(f"Correct: {item.correct_letter} ({item.correct_answer})")


if __name__ == "__main__":
    # Test loading
    print("Loading BBQ dataset...")
    items = load_bbq_items(
        categories=["ses", "age"],
        n_per_category=5,
        ambiguous_only=True,
    )
    
    print(f"\nLoaded {len(items)} items")
    print("\n" + "=" * 60)
    
    for i, item in enumerate(items[:3]):
        print(f"\n--- Item {i + 1} ---")
        print_bbq_item(item)
        print()
