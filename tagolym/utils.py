"""Supporting functions and Python classes to streamline the pipeline, 
includes:

- [X] loading and saving dictionaries,
- [X] custom encoder to convert numpy objects to JSON serializable, and
- [X] stratified data splitting algorithm for multilabel classification.
"""

import json
import itertools
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _BaseKFold
from tagolym._typing import ndarray, DataFrame, Iterator, Any, Optional, JSONEncoder, FilePath, RandomState


def load_dict(filepath: FilePath) -> dict:
    """Deserialize filepath of a JSON document to a Python object.

    Args:
        filepath (FilePath): Path of a JSON document to load from.

    Returns:
        Python dictionary.
    """
    with open(filepath) as fp:
        return json.load(fp)


def save_dict(d: dict, filepath: FilePath, cls: Optional[type[JSONEncoder]] = None) -> None:
    """Serialize a dictionary as a JSON formatted stream to a filepath.

    Args:
        d (dict): Python dictionary.
        filepath (FilePath): Path of a JSON document to save into.
        cls (Optional[type[JSONEncoder]], optional): Custom JSON encoder. 
            Defaults to None.
    """
    with open(filepath, "w") as fp:
        json.dump(d, fp=fp, cls=cls, indent=4)


def fold_tie_break(desired_samples_per_fold: ndarray, M: ndarray, random_state: Optional[RandomState] = check_random_state(None)) -> int:
    """Helper function to split a tie between folds with same desirability of 
    a given sample.

    Args:
        desired_samples_per_fold (ndarray): Number of samples desired per fold.
        M (ndarray): List of folds between which to break the tie.
        random_state (Optional[RandomState], optional): The random state seed. 
            Defaults to check_random_state(None).

    Returns:
        The selected fold index to put samples into.
    """
    if len(M) == 1:
        return M[0]
    else:
        max_val = max(desired_samples_per_fold[M])
        M_prim = np.where(np.array(desired_samples_per_fold) == max_val)[0]
        M_prim = np.array([x for x in M_prim if x in M])
        if random_state:
            if isinstance(random_state, np.random.RandomState):
                return random_state.choice(M_prim, 1)[0]
            else:
                np.random.seed(random_state)
        return np.random.choice(M_prim, 1)[0]


def get_most_desired_combination(samples_with_combination: dict[tuple, list]) -> Optional[tuple]:
    """Select the next most desired combination whose evidence should be split 
    among folds.

    Args:
        samples_with_combination (dict[tuple, list]): Mapping from each label 
            combination present in binarized labels to list of sample indices 
            that have this combination assigned.

    Returns:
        The combination to split next.
    """
    currently_chosen = None
    best_number_of_combinations, best_support_size = None, None

    for combination, evidence in samples_with_combination.items():
        number_of_combinations, support_size = (len(set(combination)), len(evidence))
        if support_size == 0:
            continue
        if currently_chosen is None or (
            best_number_of_combinations < number_of_combinations
            and best_support_size > support_size
        ):
            currently_chosen = combination
            best_number_of_combinations, best_support_size = (
                number_of_combinations,
                support_size,
            )

    return currently_chosen


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, obj: Any) -> Any:
        """Convert numpy objects to JSON serializable.

        Args:
            obj (Any): Numpy data type.

        Returns:
            Corresponding JSON serializable data type.
        """
        if isinstance(obj, (np.int_, np.intc, np.intp,
                            np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):

            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


class IterativeStratification(_BaseKFold):
    """Iteratively stratify a multilabel dataset into folds."""
    
    def __init__(
        self,
        n_splits: int = 3,
        order: int = 1,
        sample_distribution_per_fold: Optional[list[float]] = None,
        shuffle: bool = False,
        random_state: Optional[RandomState] = None,
    ) -> None:
        """Construct an interative stratifier that splits data into folds and 
        maintain balanced representation with respect to order-th label 
        combinations.

        Args:
            n_splits (int, optional): The number of folds to stratify into. 
                Defaults to 3.
            order (int, optional): The order of label relationship to take 
                into account when balancing sample distribution across labels. 
                Defaults to 1.
            sample_distribution_per_fold (Optional[list[float]], optional): 
                Desired percentage of samples in each fold. If `None`, then 
                equal distribution of samples per fold is assumed i.e. 
                `1/n_splits` for each fold. The value is held in 
                `self.percentage_per_fold`. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data before 
                splitting into batches. Note that the samples within each 
                split will not be shuffled. Defaults to False.
            random_state (Optional[RandomState], optional): Integer to seed 
                the Random Number Generator (RNG), or the RNG state to use. If 
                `None`, then the global state of numpy RNG is used. Defaults 
                to `None`.
        """
        self._rng_state = check_random_state(random_state)
        need_shuffle = shuffle or random_state is not None
        self.order = order
        super(IterativeStratification, self).__init__(
            n_splits,
            shuffle=need_shuffle,
            random_state=self._rng_state if need_shuffle else None,
        )

        if sample_distribution_per_fold:
            self.percentage_per_fold = sample_distribution_per_fold
        else:
            self.percentage_per_fold = [
                1 / float(self.n_splits) for _ in range(self.n_splits)
            ]

    def prepare_stratification(self, y: ndarray) -> tuple:
        """Prepares variables for performing stratification.

        Args:
            y (ndarray): Binarized labels.

        Returns:
            See documentation of [distribute_positive_evidence]
                [utils.IterativeStratification.distribute_positive_evidence].
        """
        self.n_samples, self.n_labels = y.shape
        self.desired_samples_per_fold = np.array(
            [self.percentage_per_fold[i] * self.n_samples for i in range(self.n_splits)]
        )
        rows = sp.lil_matrix(y).rows
        rows_used = {i: False for i in range(self.n_samples)}
        per_row_combinations = [[] for i in range(self.n_samples)]
        samples_with_combination = {}
        folds = [[] for _ in range(self.n_splits)]
        
        for sample_index, label_assignment in enumerate(rows):
            for combination in itertools.combinations_with_replacement(
                label_assignment, self.order
            ):
                if combination not in samples_with_combination:
                    samples_with_combination[combination] = []

                samples_with_combination[combination].append(sample_index)
                per_row_combinations[sample_index].append(combination)

        self.desired_samples_per_combination_per_fold = {
            combination: np.array(
                [
                    len(evidence_for_combination) * self.percentage_per_fold[j]
                    for j in range(self.n_splits)
                ]
            )
            for combination, evidence_for_combination in samples_with_combination.items()
        }
        return (
            rows_used,
            per_row_combinations,
            samples_with_combination,
            folds,
        )

    def distribute_positive_evidence(self, rows_used: dict[int, bool], folds: list[list], samples_with_combination: dict[tuple, list], per_row_combinations: list[list]) -> None:
        """Internal method to distribute evidence for labeled samples across 
        folds.

        Args:
            rows_used (dict[int, bool]): Mapping from a given sample index to 
                a boolean value indicating whether it has been already 
                assigned to a fold or not.
            folds (list[list]): List of lists to be populated with samples.
            samples_with_combination (dict[tuple, list]): Mapping from each 
                label combination present in binarized labels to list of 
                sample indices that have this combination assigned.
            per_row_combinations (list[list]): List of all label combinations 
                of order `self.order` present in binarized labels per row.
        """
        l = get_most_desired_combination(samples_with_combination)
        while l is not None:
            while len(samples_with_combination[l]) > 0:
                row = samples_with_combination[l].pop()
                if rows_used[row]:
                    continue

                max_val = max(self.desired_samples_per_combination_per_fold[l])
                M = np.where(
                    np.array(self.desired_samples_per_combination_per_fold[l])
                    == max_val
                )[0]
                m = fold_tie_break(
                    self.desired_samples_per_combination_per_fold[l], M, self._rng_state
                )
                folds[m].append(row)
                rows_used[row] = True
                for i in per_row_combinations[row]:
                    if row in samples_with_combination[i]:
                        samples_with_combination[i].remove(row)
                    self.desired_samples_per_combination_per_fold[i][m] -= 1
                self.desired_samples_per_fold[m] -= 1

            l = get_most_desired_combination(samples_with_combination)

    def distribute_negative_evidence(self, rows_used: dict[int, bool], folds: list[list]) -> None:
        """Internal method to distribute evidence for unlabeled samples across 
        folds.

        Args:
            rows_used (dict[int, bool]): Mapping from a given sample index to 
                a boolean value indicating whether it has been already 
                assigned to a fold or not.
            folds (list[list]): List of lists to be populated with samples.
        """
        available_samples = [i for i, v in rows_used.items() if not v]
        samples_left = len(available_samples)

        while samples_left > 0:
            row = available_samples.pop()
            rows_used[row] = True
            samples_left -= 1
            fold_selected = self._rng_state.choice(
                np.where(self.desired_samples_per_fold > 0)[0], 1
            )[0]
            self.desired_samples_per_fold[fold_selected] -= 1
            folds[fold_selected].append(row)

    def _iter_test_indices(self, X: DataFrame, y: ndarray, groups: Any = None) -> Iterator[list]:
        """Internal method for providing scikit-learn's split with folds.

        Args:
            X (DataFrame): Data with size `(n_samples, n_features)`.
            y (ndarray): Binarized labels on which stratification is done.
            groups (Any, optional): Always ignored, exists for compatibility. 
                Defaults to None.

        Yields:
            Iterator[list]: Indices of samples for a given fold, yielded for 
                each of the folds.
        """
        (
            rows_used,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = self.prepare_stratification(y)

        self.distribute_positive_evidence(
            rows_used, folds, samples_with_combination, per_row_combinations
        )
        self.distribute_negative_evidence(rows_used, folds)

        for fold in folds:
            yield fold