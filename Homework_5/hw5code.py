import numpy as np
from collections import Counter
from itertools import combinations


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector).ravel()
    target_vector = np.asarray(target_vector).ravel()

    if feature_vector.size <= 1 or np.all(feature_vector == feature_vector[0]):
        return np.array([]), np.array([]), None, None

    order = np.argsort(feature_vector)
    feature_sorted = feature_vector[order]

    thresholds = (feature_sorted[1:] + feature_sorted[:-1]) / 2.0
    valid_split_mask = feature_sorted[1:] != feature_sorted[:-1]
    if not np.any(valid_split_mask):
        return np.array([]), np.array([]), None, None

    classes, inverse = np.unique(target_vector, return_inverse=True)
    if classes.size != 2:
        raise ValueError("find_best_split supports only binary classification")

    target_sorted = inverse[order]
    total_count = target_vector.size

    cumulative_positive = np.cumsum(target_sorted[:-1] == 1)
    left_count = np.arange(1, target_sorted.size)
    right_count = total_count - left_count

    left_positive = cumulative_positive
    right_positive = np.sum(inverse == 1) - left_positive

    left_negative = left_count - left_positive
    right_negative = right_count - right_positive

    left_gini = 1.0 - (left_positive / left_count) ** 2 - (left_negative / left_count) ** 2
    right_gini = 1.0 - (right_positive / right_count) ** 2 - (right_negative / right_count) ** 2
    ginis = -(left_count / total_count) * left_gini - (right_count / total_count) * right_gini
    thresholds = thresholds[valid_split_mask]
    ginis = ginis[valid_split_mask]

    max_gini = np.max(ginis)
    candidate_indices = np.flatnonzero(ginis == max_gini)
    best_index = candidate_indices[np.argmin(thresholds[candidate_indices])]
    return thresholds, ginis, thresholds[best_index], ginis[best_index]


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if sub_y.size == 0:
            node["type"] = "terminal"
            node["class"] = None
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        current_depth = node.get("depth", 0)

        if self._max_depth is not None and current_depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and sub_y.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Map labels to 0/1 once to make gini safe for non-integer targets.
        _, y_encoded = np.unique(sub_y, return_inverse=True)

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                _, _, threshold, gini = find_best_split(feature_vector, y_encoded)

                if threshold is None:
                    continue

                split = feature_vector <= threshold
                if split.sum() < (self._min_samples_leaf or 1):
                    continue
                if (~split).sum() < (self._min_samples_leaf or 1):
                    continue

                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector <= threshold
                    threshold_best = threshold

            elif feature_type == "categorical":
                feature_vector = sub_X[:, feature]
                categories = np.unique(feature_vector)
                if categories.size <= 1:
                    continue

                # Order categories by target mean, then map to ranks for splitting.
                category_means = {
                    category: y_encoded[feature_vector == category].mean()
                    for category in categories
                }
                ordered_categories = [
                    category
                    for category, _ in sorted(
                        category_means.items(),
                        key=lambda item: (item[1], item[0]),
                    )
                ]
                category_rank = {category: idx for idx, category in enumerate(ordered_categories)}
                mapped_feature = np.array(
                    [category_rank[value] for value in feature_vector],
                    dtype=float,
                )

                _, _, threshold, gini = find_best_split(mapped_feature, y_encoded)
                if threshold is None:
                    continue

                split = mapped_feature <= threshold
                categories_left = [
                    category
                    for category in ordered_categories
                    if category_rank[category] <= threshold
                ]
                if split.sum() < (self._min_samples_leaf or 1):
                    continue
                if (~split).sum() < (self._min_samples_leaf or 1):
                    continue

                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = mapped_feature <= threshold
                    threshold_best = categories_left
            else:
                raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": current_depth + 1}, {"depth": current_depth + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            go_left = x[feature] <= node["threshold"]
        elif self._feature_types[feature] == "categorical":
            go_left = x[feature] in node["categories_split"]
        else:
            raise ValueError

        next_node = node["left_child"] if go_left else node["right_child"]
        return self._predict_node(x, next_node)

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
