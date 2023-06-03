import json
import itertools
import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _BaseKFold


def load_dict(filepath):
    with open(filepath) as fp:
        return json.load(fp)


def save_dict(d, filepath, cls=None):
    with open(filepath, "w") as fp:
        json.dump(d, fp=fp, cls=cls, indent=4)


def _fold_tie_break(desired_samples_per_fold, M, random_state=check_random_state(None)):
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


def _get_most_desired_combination(samples_with_combination):
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
    def default(self, obj):
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
    def __init__(
        self,
        n_splits=3,
        order=1,
        sample_distribution_per_fold=None,
        shuffle=False,
        random_state=None,
    ):
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

    def _prepare_stratification(self, y):
        self.n_samples, self.n_labels = y.shape
        self.desired_samples_per_fold = np.array(
            [self.percentage_per_fold[i] * self.n_samples for i in range(self.n_splits)]
        )
        rows = sp.lil_matrix(y).rows
        rows_used = {i: False for i in range(self.n_samples)}
        all_combinations = []
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
                all_combinations.append(combination)
                per_row_combinations[sample_index].append(combination)

        all_combinations = [list(x) for x in set(all_combinations)]

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

    def _distribute_positive_evidence(self, rows_used, folds, samples_with_combination, per_row_combinations):
        l = _get_most_desired_combination(samples_with_combination)
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
                m = _fold_tie_break(
                    self.desired_samples_per_combination_per_fold[l], M, self._rng_state
                )
                folds[m].append(row)
                rows_used[row] = True
                for i in per_row_combinations[row]:
                    if row in samples_with_combination[i]:
                        samples_with_combination[i].remove(row)
                    self.desired_samples_per_combination_per_fold[i][m] -= 1
                self.desired_samples_per_fold[m] -= 1

            l = _get_most_desired_combination(samples_with_combination)

    def _distribute_negative_evidence(self, rows_used, folds):
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

    def _iter_test_indices(self, X, y=None, groups=None):
        (
            rows_used,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = self._prepare_stratification(y)

        self._distribute_positive_evidence(
            rows_used, folds, samples_with_combination, per_row_combinations
        )
        self._distribute_negative_evidence(rows_used, folds)

        for fold in folds:
            yield fold