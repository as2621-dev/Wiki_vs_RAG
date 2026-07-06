"""Frozen, checksum-locked question bank across three tiers (lookup / relational / timeline).

Each question targets exactly one anchor work (a novel or a movie script); the harness only
runs a question at corpus sizes where that work is loaded (source order is defined in
``corpus.py``). Gold answers are reference text for the LLM judge, not exact-match strings.

The bank is **frozen**: ``QUESTION_BANK_CHECKSUM`` is a stable SHA-256 over the canonical
serialization of every question. ``verify_question_bank_checksum`` guards against any accidental
edit to a ``question_text`` / ``gold_answer`` / ``tier`` silently changing a published result.
"""

import hashlib
import json
from collections.abc import Iterable

from models import BenchmarkQuestion, QuestionTier

# Reason: novels come from Gutenberg (``corpus.GUTENBERG_BOOKS``); movies are sourced by the
# corpus movie-script path (M1). book_key values here are the anchor keys the corpus loader must
# also produce — content coordination, not a code dependency.
QUESTION_BANK: list[BenchmarkQuestion] = [
    # ─── Sherlock Holmes (novel) ──────────────────────────────
    BenchmarkQuestion(
        question_id="holmes_lookup_1",
        book_key="holmes",
        tier=QuestionTier.LOOKUP,
        question_text="What is Sherlock Holmes's profession?",
        gold_answer="A consulting detective.",
    ),
    BenchmarkQuestion(
        question_id="holmes_relational_1",
        book_key="holmes",
        tier=QuestionTier.RELATIONAL,
        question_text="How is Dr. Watson connected to Sherlock Holmes?",
        gold_answer=(
            "Watson is Holmes's friend and flatmate at 221B Baker Street, and the narrator who "
            "chronicles his cases."
        ),
    ),
    BenchmarkQuestion(
        question_id="holmes_timeline_1",
        book_key="holmes",
        tier=QuestionTier.TIMELINE,
        question_text=(
            "In 'A Scandal in Bohemia', how does Holmes try to locate the photograph, and what "
            "happens the next morning?"
        ),
        gold_answer=(
            "Holmes stages a fake fire alarm to make Irene Adler instinctively reveal the "
            "photograph's hiding place; but by the next morning she has seen through him and fled "
            "with it, leaving behind a decoy photograph and a letter."
        ),
    ),
    # ─── Dracula (novel) ──────────────────────────────────────
    BenchmarkQuestion(
        question_id="dracula_lookup_1",
        book_key="dracula",
        tier=QuestionTier.LOOKUP,
        question_text="Who leads the group that hunts Dracula?",
        gold_answer="Professor Abraham Van Helsing.",
    ),
    BenchmarkQuestion(
        question_id="dracula_relational_1",
        book_key="dracula",
        tier=QuestionTier.RELATIONAL,
        question_text="Who are Lucy Westenra's three suitors?",
        gold_answer=(
            "Dr. John Seward, Quincey Morris, and Arthur Holmwood (who becomes her fiancé)."
        ),
    ),
    BenchmarkQuestion(
        question_id="dracula_timeline_1",
        book_key="dracula",
        tier=QuestionTier.TIMELINE,
        question_text="What sequence of events transforms Lucy Westenra after Dracula reaches England?",
        gold_answer=(
            "Dracula arrives at Whitby aboard the ship Demeter and preys on Lucy; she sickens and "
            "sleepwalks, receives blood transfusions, dies, rises as a vampire (the 'Bloofer Lady'), "
            "and is finally staked by the group."
        ),
    ),
    # ─── The Count of Monte Cristo (novel) ────────────────────
    BenchmarkQuestion(
        question_id="monte_cristo_lookup_1",
        book_key="monte_cristo",
        tier=QuestionTier.LOOKUP,
        question_text="What is Edmond Dantès's occupation at the start of the novel?",
        gold_answer="A sailor and first mate aboard the merchant ship Pharaon.",
    ),
    BenchmarkQuestion(
        question_id="monte_cristo_relational_1",
        book_key="monte_cristo",
        tier=QuestionTier.RELATIONAL,
        question_text="Which men conspire to have Edmond Dantès imprisoned?",
        gold_answer=(
            "Danglars and Fernand Mondego frame him with Caderousse's complicity, and the "
            "magistrate Villefort completes his imprisonment to protect himself."
        ),
    ),
    BenchmarkQuestion(
        question_id="monte_cristo_timeline_1",
        book_key="monte_cristo",
        tier=QuestionTier.TIMELINE,
        question_text="Trace how Edmond Dantès becomes the Count of Monte Cristo.",
        gold_answer=(
            "Falsely accused and imprisoned in the Château d'If, he befriends the Abbé Faria, who "
            "educates him and reveals the Monte Cristo treasure; Dantès escapes, recovers the "
            "treasure, and reinvents himself as the wealthy Count to pursue revenge."
        ),
    ),
    # ─── Les Misérables (novel) ───────────────────────────────
    BenchmarkQuestion(
        question_id="les_mis_lookup_1",
        book_key="les_mis",
        tier=QuestionTier.LOOKUP,
        question_text="Which police inspector relentlessly pursues Jean Valjean?",
        gold_answer="Inspector Javert.",
    ),
    BenchmarkQuestion(
        question_id="les_mis_relational_1",
        book_key="les_mis",
        tier=QuestionTier.RELATIONAL,
        question_text="What is Jean Valjean's relationship to Cosette?",
        gold_answer=(
            "He becomes her adoptive father after promising her dying mother, Fantine, to care for "
            "her."
        ),
    ),
    BenchmarkQuestion(
        question_id="les_mis_timeline_1",
        book_key="les_mis",
        tier=QuestionTier.TIMELINE,
        question_text="What chain of events sets Jean Valjean on his path after leaving prison?",
        gold_answer=(
            "Released after 19 years imprisoned for stealing bread, he is shown mercy by Bishop "
            "Myriel (who lets him keep the stolen silver); this inspires him to reform, assume a "
            "new identity, and become a factory owner and mayor."
        ),
    ),
    # ─── War and Peace (novel) ────────────────────────────────
    BenchmarkQuestion(
        question_id="war_and_peace_lookup_1",
        book_key="war_and_peace",
        tier=QuestionTier.LOOKUP,
        question_text="What historical conflict forms the backdrop of the novel?",
        gold_answer="The Napoleonic Wars, especially Napoleon's 1812 invasion of Russia.",
    ),
    BenchmarkQuestion(
        question_id="war_and_peace_relational_1",
        book_key="war_and_peace",
        tier=QuestionTier.RELATIONAL,
        question_text="How are Pierre Bezukhov and Natasha Rostova connected by the end of the novel?",
        gold_answer="Pierre marries Natasha.",
    ),
    BenchmarkQuestion(
        question_id="war_and_peace_timeline_1",
        book_key="war_and_peace",
        tier=QuestionTier.TIMELINE,
        question_text="What happens to Moscow during Napoleon's 1812 invasion?",
        gold_answer=(
            "Napoleon's army occupies a largely abandoned Moscow, which is swept by fire, and the "
            "French are then forced into a catastrophic winter retreat."
        ),
    ),
    # ─── The Godfather (movie) ────────────────────────────────
    BenchmarkQuestion(
        question_id="godfather_lookup_1",
        book_key="godfather",
        tier=QuestionTier.LOOKUP,
        question_text="Which of Vito Corleone's sons ultimately becomes the new Don of the family?",
        gold_answer="Michael Corleone, the youngest son.",
    ),
    BenchmarkQuestion(
        question_id="godfather_relational_1",
        book_key="godfather",
        tier=QuestionTier.RELATIONAL,
        question_text="How is Michael Corleone related to Sonny Corleone?",
        gold_answer=(
            "They are brothers, both sons of Vito Corleone; Sonny (Santino) is the hot-headed "
            "eldest son and Michael is the youngest."
        ),
    ),
    BenchmarkQuestion(
        question_id="godfather_timeline_1",
        book_key="godfather",
        tier=QuestionTier.TIMELINE,
        question_text="How does Michael Corleone first kill for the family, and what does he do afterward?",
        gold_answer=(
            "After the attempted assassination of his father, Michael volunteers to murder the "
            "rival Sollozzo and the corrupt police captain McCluskey; he shoots them during a "
            "restaurant meeting and then flees to hide in Sicily."
        ),
    ),
    # ─── Casablanca (movie) ───────────────────────────────────
    BenchmarkQuestion(
        question_id="casablanca_lookup_1",
        book_key="casablanca",
        tier=QuestionTier.LOOKUP,
        question_text="What is the name of Rick Blaine's nightclub in Casablanca?",
        gold_answer="Rick's Café Américain.",
    ),
    BenchmarkQuestion(
        question_id="casablanca_relational_1",
        book_key="casablanca",
        tier=QuestionTier.RELATIONAL,
        question_text="What is the relationship between Rick Blaine and Ilsa Lund?",
        gold_answer=(
            "They are former lovers who had a romance in Paris before Ilsa left him without "
            "explanation; she is married to the resistance leader Victor Laszlo."
        ),
    ),
    BenchmarkQuestion(
        question_id="casablanca_timeline_1",
        book_key="casablanca",
        tier=QuestionTier.TIMELINE,
        question_text="How does Rick help Ilsa and Laszlo escape at the end of the film?",
        gold_answer=(
            "Rick gives the letters of transit to Ilsa and Laszlo and sends them off on the plane "
            "to Lisbon; when Major Strasser tries to stop it, Rick shoots him, and Rick stays "
            "behind with Captain Renault."
        ),
    ),
    # ─── Pulp Fiction (movie) ─────────────────────────────────
    BenchmarkQuestion(
        question_id="pulp_fiction_lookup_1",
        book_key="pulp_fiction",
        tier=QuestionTier.LOOKUP,
        question_text="What is the profession of Vincent Vega and Jules Winnfield?",
        gold_answer="They are hitmen working for the crime boss Marsellus Wallace.",
    ),
    BenchmarkQuestion(
        question_id="pulp_fiction_relational_1",
        book_key="pulp_fiction",
        tier=QuestionTier.RELATIONAL,
        question_text="Who is Mia Wallace in relation to Marsellus Wallace?",
        gold_answer=(
            "Mia is Marsellus Wallace's wife; Vincent is asked to entertain her while Marsellus "
            "is out of town."
        ),
    ),
    BenchmarkQuestion(
        question_id="pulp_fiction_timeline_1",
        book_key="pulp_fiction",
        tier=QuestionTier.TIMELINE,
        question_text="What happens after Mia Wallace mistakes Vincent's heroin for cocaine?",
        gold_answer=(
            "Mia snorts the heroin and overdoses; Vincent rushes her to Lance's house, where they "
            "revive her by injecting a shot of adrenaline directly into her heart."
        ),
    ),
    # ─── The Wizard of Oz (movie) ─────────────────────────────
    BenchmarkQuestion(
        question_id="wizard_of_oz_lookup_1",
        book_key="wizard_of_oz",
        tier=QuestionTier.LOOKUP,
        question_text="What is the name of Dorothy's dog?",
        gold_answer="Toto.",
    ),
    BenchmarkQuestion(
        question_id="wizard_of_oz_relational_1",
        book_key="wizard_of_oz",
        tier=QuestionTier.RELATIONAL,
        question_text="Which three companions join Dorothy on her journey to the Emerald City, and what does each want?",
        gold_answer=(
            "The Scarecrow (who wants a brain), the Tin Man (who wants a heart), and the Cowardly "
            "Lion (who wants courage)."
        ),
    ),
    BenchmarkQuestion(
        question_id="wizard_of_oz_timeline_1",
        book_key="wizard_of_oz",
        tier=QuestionTier.TIMELINE,
        question_text="How does Dorothy ultimately return home to Kansas?",
        gold_answer=(
            "After melting the Wicked Witch and exposing the Wizard as a fraud, Glinda tells "
            "Dorothy her ruby slippers can carry her home; Dorothy taps her heels together "
            "repeating 'There's no place like home.'"
        ),
    ),
    # ─── Star Wars: A New Hope (movie) ────────────────────────
    BenchmarkQuestion(
        question_id="star_wars_lookup_1",
        book_key="star_wars",
        tier=QuestionTier.LOOKUP,
        question_text="What is the name of the Empire's planet-destroying space station?",
        gold_answer="The Death Star.",
    ),
    BenchmarkQuestion(
        question_id="star_wars_relational_1",
        book_key="star_wars",
        tier=QuestionTier.RELATIONAL,
        question_text="What is Obi-Wan Kenobi's relationship to Luke Skywalker in the film?",
        gold_answer=(
            "Obi-Wan is an old Jedi Knight who served with Luke's father; he becomes Luke's mentor "
            "and begins training him in the ways of the Force."
        ),
    ),
    BenchmarkQuestion(
        question_id="star_wars_timeline_1",
        book_key="star_wars",
        tier=QuestionTier.TIMELINE,
        question_text="How is the Death Star destroyed at the end of the film?",
        gold_answer=(
            "Luke Skywalker, trusting the Force and Obi-Wan's guiding voice, fires proton "
            "torpedoes into a small thermal exhaust port, triggering a chain reaction that "
            "destroys the Death Star."
        ),
    ),
    # ─── The Shawshank Redemption (movie) ─────────────────────
    BenchmarkQuestion(
        question_id="shawshank_lookup_1",
        book_key="shawshank",
        tier=QuestionTier.LOOKUP,
        question_text="For what crime is Andy Dufresne sentenced to Shawshank prison?",
        gold_answer=(
            "For the murder of his wife and her lover — a crime he maintains he did not commit."
        ),
    ),
    BenchmarkQuestion(
        question_id="shawshank_relational_1",
        book_key="shawshank",
        tier=QuestionTier.RELATIONAL,
        question_text="Who is Andy Dufresne's closest friend inside Shawshank?",
        gold_answer=(
            "Ellis 'Red' Redding, the long-term inmate who narrates the story and can procure "
            "contraband."
        ),
    ),
    BenchmarkQuestion(
        question_id="shawshank_timeline_1",
        book_key="shawshank",
        tier=QuestionTier.TIMELINE,
        question_text="How does Andy Dufresne escape from Shawshank?",
        gold_answer=(
            "Over roughly two decades he tunnels through his cell wall with a small rock hammer, "
            "hiding the hole behind a poster; one stormy night he crawls out through a sewage pipe "
            "to freedom and later collects the warden's laundered money."
        ),
    ),
]

# Reason: fields hashed are exactly the ones a published result depends on. question_id anchors
# each record so reordering the bank can never change the checksum; content fields (book_key,
# tier, question_text, gold_answer) are what an accidental edit would alter.
_CHECKSUM_FIELDS: tuple[str, ...] = ("question_id", "book_key", "tier", "question_text", "gold_answer")


def compute_question_bank_checksum(bank: list[BenchmarkQuestion] = QUESTION_BANK) -> str:
    """Compute a stable SHA-256 checksum over the frozen question bank.

    Canonicalization is deterministic across runs and Python versions: records are sorted by
    ``question_id`` and serialized as JSON with sorted keys, ASCII escaping, and no volatile
    whitespace. Editing any hashed field of any question changes the digest.

    Args:
        bank: The questions to hash; defaults to the module-level ``QUESTION_BANK``.

    Returns:
        The hex SHA-256 digest of the canonical serialization.

    Example:
        >>> compute_question_bank_checksum() == QUESTION_BANK_CHECKSUM
        True
    """
    records = sorted(
        ({field: _hashable_field(question, field) for field in _CHECKSUM_FIELDS} for question in bank),
        key=lambda record: record["question_id"],
    )
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hashable_field(question: BenchmarkQuestion, field: str) -> str:
    """Return the string form of a question field for checksumming (enum → its stable value)."""
    value = getattr(question, field)
    return value.value if isinstance(value, QuestionTier) else value


# Frozen digest of the bank above. Regenerate deliberately (and review the diff) only when the
# golden set is intentionally re-frozen: `python -c "import questions; print(questions.compute_question_bank_checksum())"`.
QUESTION_BANK_CHECKSUM: str = "c4fb5f26f38137ef27df51af12181daf164aab737be1161f45571bdc8ec40d76"


def verify_question_bank_checksum() -> bool:
    """Return True iff the live ``QUESTION_BANK`` still matches the frozen checksum.

    A False result means a ``question_text`` / ``gold_answer`` / ``tier`` (or any hashed field)
    was edited without deliberately re-freezing the bank — a guard against silently changing a
    published result (PRD story #2).

    Example:
        >>> verify_question_bank_checksum()
        True
    """
    return compute_question_bank_checksum(QUESTION_BANK) == QUESTION_BANK_CHECKSUM


def questions_for_corpus_size(corpus_size: int, book_order: list[str]) -> list[BenchmarkQuestion]:
    """Return questions whose anchor work is loaded at the given book-count size.

    ``corpus_size`` is the internal fixed-ordering detail (book/script count); the true sweep
    axis is token count — see ``questions_for_token_target``.

    Args:
        corpus_size: Number of works loaded (1..len(book_order)).
        book_order: Fixed anchor-key order (e.g. from ``corpus.GUTENBERG_BOOKS``).

    Returns:
        Questions answerable at this corpus size.

    Example:
        >>> qs = questions_for_corpus_size(1, ["holmes", "dracula"])
        >>> {q.book_key for q in qs}
        {'holmes'}
    """
    return questions_for_token_target(book_order[:corpus_size])


def questions_for_token_target(loaded_source_keys: Iterable[str]) -> list[BenchmarkQuestion]:
    """Return the questions answerable given the works the corpus loaded for a token target.

    The corpus's cumulative loader decides which anchor works fit under a token target and
    yields them in fixed source order; their keys are **injected** here. A question is returned
    only if its ``book_key`` is among the loaded works — a question whose anchor is not yet
    loaded at that target is never returned (PRD decision #3).

    Args:
        loaded_source_keys: Anchor keys loaded at the token target, from the corpus loader
            (``corpus.GUTENBERG_BOOKS`` order today; movie keys added in M1). Never hard-coded
            to a book count here.

    Returns:
        Questions whose anchor work is loaded, in bank order.

    Example:
        >>> qs = questions_for_token_target(["holmes"])
        >>> {q.book_key for q in qs}
        {'holmes'}
    """
    loaded = set(loaded_source_keys)
    return [question for question in QUESTION_BANK if question.book_key in loaded]
