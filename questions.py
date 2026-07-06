"""Fixed question bank across three tiers (lookup / relational / timeline).

Each question targets exactly one book; the harness only runs a question at
corpus sizes where that book is loaded (book order is defined in ``corpus.py``).
Gold answers are reference text for the LLM judge, not exact-match strings.
"""

from models import BenchmarkQuestion, QuestionTier

QUESTION_BANK: list[BenchmarkQuestion] = [
    # ─── Sherlock Holmes (loaded at corpus size >= 1) ─────────
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
    # ─── Dracula (corpus size >= 2) ───────────────────────────
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
    # ─── The Count of Monte Cristo (corpus size >= 3) ─────────
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
    # ─── Les Misérables (corpus size >= 4) ────────────────────
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
    # ─── War and Peace (corpus size >= 5) ─────────────────────
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
]


def questions_for_corpus_size(corpus_size: int, book_order: list[str]) -> list[BenchmarkQuestion]:
    """Return questions whose target book is loaded at the given corpus size.

    Args:
        corpus_size: Number of books loaded (1..len(book_order)).
        book_order: Fixed book_key order (from ``corpus.GUTENBERG_BOOKS``).

    Returns:
        Questions answerable at this corpus size.

    Example:
        >>> qs = questions_for_corpus_size(1, ["holmes", "dracula"])
        >>> {q.book_key for q in qs}
        {'holmes'}
    """
    loaded = set(book_order[:corpus_size])
    return [q for q in QUESTION_BANK if q.book_key in loaded]
