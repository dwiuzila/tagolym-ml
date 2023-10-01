"""All functions regarding data are written in this module, including data 
split, preprocessing, and transformation.

Definitions:
    | Term  | Definition                                               |
    | ----- | ---------------------------------------------------------|
    | Post  | String explaining a math problem written in LaTeX.       |
    | Token | Preprocessed post.                                       |
    | Tag   | User input string suggesting in what category a post is. \
              A post could have multiple tags.                         |
    | Label | Preprocessed tag. Only 10 labels are defined.            |
"""

import regex as re
from collections import defaultdict
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from tagolym._typing import ndarray, Series, DataFrame, Iterable, Optional, Union, Transformer, RandomState

from config import config
from tagolym import utils


def create_tag_mapping(tags: Series) -> defaultdict[str, list]:
    """Create a dictionary in which each key is a tag and each value is a 
    sublist of complete labels. The mapping is defined if the lowercased tag 
    contains an element of partial labels as its substring.
    
    Partial labels are defined as 
    ```python
    ["algebra", "geometr", "number theor", "combinator", "inequalit", 
     "function", "polynomial", "circle", "trigonometr", "modul"]
    ```
    and complete labels are defined as
    ```python
    ["algebra", "geometry", "number theory", "combinatorics", "inequality", 
     "function", "polynomial", "circle", "trigonometry", "modular arithmetic"]
    ```

    For example, the tag `["combinatorial geometry"]` will give a key-value 
    pair `{"combinatorial geometry": ["combinatorics", "geometry"]}`.

    Args:
        tags (Series): Collection of list of tags annotated by users.

    Returns:
        Mapping from tag to sublist of complete labels.
    """
    mappings = []
    for plb, clb in zip(config.PARTIAL_LABELS, config.COMPLETE_LABELS):
        similar_tags = set([t.lower() for tag in tags for t in tag if plb in t.lower()])
        mappings.append({tag: clb for tag in similar_tags})

    mapping = defaultdict(list)
    for mpg in mappings:
        for key, value in mpg.items():
            mapping[key].append(value)
    
    return mapping


def preprocess_tag(x: list, mapping: defaultdict[str, list]) -> list:
    """Preprocess a list of tags, including: lowercasing, mapping to complete 
    labels, dropping duplicates, and sorting.

    Args:
        x (list): List of tags annotated by users.
        mapping (defaultdict[str, list]): Mapping from tag to sublist of 
            complete labels.

    Returns:
        Preprocessed list of tags.
    """
    x = [tag.lower() for tag in x]       # lowercase all
    x = map(mapping.get, x)              # map tags
    x = filter(None, x)                  # remove None
    x = [t for tag in x for t in tag]    # flattened tags
    x = sorted(list(set(x)))             # drop duplicates and sort
    return x


def extract_features(equation_pattern: str, x: str) -> str:
    r"""Extract LaTeX commands inside math modes from a given text.

    For example, this render
    > Find all functions $f:(0,\infty)\rightarrow (0,\infty)$ such that for 
    > any $x,y\in (0,\infty)$, 
    > $$
    > xf(x^2)f(f(y)) + f(yf(x)) = f(xy) \left(f(f(x^2)) + f(f(y^2))\right).
    > $$
    
    will become
    > Find all functions  \infty \infty  such that for any  \in \infty ,  \left

    Args:
        equation_pattern (str): Regex pattern for finding math modes.
        x (str): Input text written in LaTeX.

    Returns:
        Text with extracted LaTeX commands.
    """
    pattern = re.findall(equation_pattern, x)
    ptn_len = [len(ptn) for ptn in pattern]
    pattern = ["".join(ptn) for ptn in pattern]
    syntax = [" ".join(re.findall(r"\\(?:[^a-zA-Z]|[a-zA-Z]+[*=']?)", ptn)) for ptn in pattern]
    split = ["" if s is None else s for s in re.split(equation_pattern, x)]

    i = 0
    for ptn, length, cmd in zip(pattern, ptn_len, syntax):
        while "".join(split[i : i + length]) != ptn:
            i += 1
        split[i : i + length] = [cmd]

    return " ".join(split)


def preprocess_post(x: str, nocommand: bool = False, stem: bool = False) -> str:
    """Deep clean a post, using [extract_features][data.extract_features] as 
    one of the steps.

    Args:
        x (str): Post written in LaTeX.
        nocommand (bool, optional): Whether to remove command words, i.e. 
            `["prove", "let", "find", "show", "given"]`.
        stem (bool, optional): Whether to apply word stemming.

    Returns:
        Cleaned post.
    """
    x = x.lower()                                       # lowercase all
    x = re.sub(r"http\S+", "", x)                       # remove URLs
    x = x.replace("$$$", "$$ $")                        # separate triple dollars
    x = x.replace("\n", " ")                            # remove new lines
    x = extract_features(config.EQUATION_PATTERN, x)    # extract latex
    x = re.sub(config.ASYMPTOTE_PATTERN, "", x)         # remove asymptote

    # remove stopwords
    x = x.replace("\\", " \\")
    x = " ".join(word for word in x.split() if word not in config.STOPWORDS)

    x = re.sub(r"([-;.,!?<=>])", r" \1 ", x)            # separate filters from words
    x = re.sub("[^A-Za-z0-9]+", " ", x)                 # remove non-alphanumeric chars

    # clean command words
    if nocommand:
        x = " ".join(word for word in x.split() if word not in config.COMMANDS)

    # stem words
    if stem:
        stemmer = PorterStemmer()
        x = " ".join(stemmer.stem(word) for word in x.split())
    
    # remove spaces at the beginning and end
    x = x.strip()

    return x


def preprocess(df: DataFrame, nocommand: bool, stem: bool) -> DataFrame:
    """End-to-end data preprocessing on all posts and their corresponding 
    tags, then drop all data points with an empty preprocessed post afterward.

    Args:
        df (DataFrame): Raw data containing posts and their corresponding tags.
        nocommand (bool): Whether to remove command words, i.e. `["prove", 
            "let", "find", "show", "given"]`.
        stem (bool): Whether to apply word stemming.

    Returns:
        Preprocessed data used for modeling.
    """
    mapping = create_tag_mapping(df["tags"])
    df["token"] = df["post_canonical"].apply(preprocess_post, args=(nocommand, stem))
    df["tags"] = df["tags"].apply(preprocess_tag, args=(mapping,))
    df = df[df["token"] != ""].reset_index(drop=True)
    return df


def binarize(labels: Series) -> tuple[ndarray, Transformer]:
    """Convert labels into a binary matrix of size `(n_samples, n_labels)` 
    indicating the presence of a complete label. For example, the labels 
    `["algebra", "inequality"]` will be transformed into `[1, 0, 0, 0, 0, 1, 
    0, 0, 0, 0]`. Besides returning the transformed labels, it also returns 
    the `MultiLabelBinarizer` object used later in downstream processes for 
    converting the matrix back to labels.

    Args:
        labels (Series): Collection of list of preprocessed tags.

    Returns:
        label_indicator: Binary matrix representation of `labels`.
        mlb: Transformer that converts `labels` to `label_indicator`.
    """
    mlb = MultiLabelBinarizer()
    label_indicator = mlb.fit_transform(labels)
    return label_indicator, mlb


def split_data(X: DataFrame, y: ndarray, train_size: float = 0.7, random_state: Optional[RandomState] = None) -> Iterable[Union[DataFrame, ndarray]]:
    """Using [utils.IterativeStratification][], split the tokens and their 
    corresponding labels into 3 parts with (customizable) 70/15/15 
    proportions, each respectively for model training, validation, and testing.

    Args:
        X (DataFrame): Preprocessed posts.
        y (ndarray): Binarized labels.
        train_size (float, optional): Fraction of training data. Defaults to 
            0.7.
        random_state (Optional[RandomState], optional): Controls the shuffling 
            applied to the data before applying the split. Pass an int for 
            reproducible output across multiple function calls. Defaults to 
            None.

    Returns:
        Tuple containing train-validation-test split of tokens and labels.
    """
    stratifier = utils.IterativeStratification(
        n_splits=3,
        order=2,
        sample_distribution_per_fold=[train_size, (1-train_size)/2, (1-train_size)/2],
        random_state=random_state,
    )

    indices = []
    for _, idx in stratifier.split(X, y):
        indices.append(idx.tolist())

    X_train, y_train = X.iloc[indices[0]], y[indices[0]]
    X_val, y_val = X.iloc[indices[1]], y[indices[1]]
    X_test, y_test = X.iloc[indices[2]], y[indices[2]]

    return X_train, X_val, X_test, y_train, y_val, y_test