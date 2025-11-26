import numpy as np
from typing import List, Tuple, Dict


def _rank_confused_pairs(cm: np.ndarray) -> List[Tuple[Tuple[int, int], int]]:
    """
    Rank pairs of digits by their symmetric misclassification count for the baseline model:
    cm[i, j] + cm[j, i].

    Parameters
    ----------
    cm : np.ndarray
        A square confusion matrix with shape (10, 10).

    Returns
    -------
    List[Tuple[Tuple[int, int], int]]
        Sorted list of ((i, j), total_misclassifications) pairs.
    """
    n = cm.shape[0]
    pairs: Dict[Tuple[int, int], int] = {}

    for i in range(n):
        for j in range(i + 1, n):
            pairs[(i, j)] = int(cm[i, j] + cm[j, i])

    return sorted(pairs.items(), key=lambda item: item[1], reverse=True)


def summarize_confusions(cm: np.ndarray) -> None:
    """
    Print detailed statistics about confusion pairs and global error metrics.
    """
    results = _rank_confused_pairs(cm)

    print("Confused Digit Pairs (sorted):")
    for (i, j), total in results:
        print(f"  {i} ↔ {j}: {total}")

    total_errors = int(cm.sum() - np.trace(cm))
    top_pair, top_val = results[0]
    top3_total = sum(val for _, val in results[:3])

    print("\n--- Summary ---")
    print(f"Total misclassifications: {total_errors}")
    print(f"Top pair {top_pair[0]}↔{top_pair[1]}: "
          f"{top_val}  ({top_val / total_errors:.2%})")
    print(f"Top 3 pairs total: {top3_total}  ({top3_total / total_errors:.2%})")


if __name__ == "__main__":
    cm = np.array([
        [961, 0, 1, 1, 0, 5, 6, 3, 2, 1],
        [0, 1110, 4, 3, 0, 1, 5, 1, 11, 0],
        [8, 11, 907, 20, 11, 7, 15, 13, 38, 2],
        [5, 0, 19, 917, 2, 23, 5, 12, 19, 8],
        [1, 4, 6, 4, 916, 0, 7, 3, 5, 36],
        [11, 1, 0, 36, 12, 767, 18, 9, 30, 8],
        [9, 4, 9, 1, 5, 19, 909, 0, 2, 0],
        [2, 7, 25, 7, 7, 2, 1, 943, 4, 30],
        [8, 12, 8, 19, 15, 34, 12, 11, 843, 12],
        [6, 8, 2, 16, 30, 12, 0, 27, 12, 896],
    ])

    summarize_confusions(cm)
