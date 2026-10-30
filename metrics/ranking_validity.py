from __future__ import annotations


def is_exact_permutation(ranking: list[str], expected_labels: set[str]) -> bool:
    """Exactly |labels| entries and each label appears once."""
    n = len(expected_labels)
    if len(ranking) != n:
        return False
    return set(ranking) == expected_labels


def degenerate_numeric_pattern_reason(ranking: list[str], n: int) -> str | None:
    """
    If labels are \"1\"…\"n\", return a reason string for excluded templates, else None.
    Precondition: ranking is an exact permutation of 1..n strings.
    """
    keys = {str(i) for i in range(1, n + 1)}
    if set(ranking) != keys:
        return None

    if ranking == [str(i) for i in range(1, n + 1)]:
        return "ascending_1_to_n"
    if ranking == [str(i) for i in range(n, 0, -1)]:
        return "descending_n_to_1"

    if n % 2 != 0:
        return None
    half = n // 2
    evens = {str(i) for i in range(2, n + 1, 2)}
    odds = {str(i) for i in range(1, n + 1, 2)}
    first, second = set(ranking[:half]), set(ranking[half:])
    if first == evens and second == odds:
        return "parity_halves_evens_then_odds"
    if first == odds and second == evens:
        return "parity_halves_odds_then_evens"
    return None


def degenerate_pattern_reason(ranking: list[str], expected_labels: set[str]) -> str | None:
    """
    None  -> not a known excluded template (count in metrics).
    str   -> excluded template (do not count hits unless --include-degenerate).
    """
    if not is_exact_permutation(ranking, expected_labels):
        return None

    numeric_expected = {str(i) for i in range(1, len(expected_labels) + 1)}
    if expected_labels == numeric_expected:
        return degenerate_numeric_pattern_reason(ranking, len(expected_labels))
    return None


def is_good_ranking(ranking: list[str], expected_labels: set[str]) -> bool:
    """Valid permutation and not an excluded numeric template (for 1…n labels)."""
    if not is_exact_permutation(ranking, expected_labels):
        return False
    return degenerate_pattern_reason(ranking, expected_labels) is None
