import regex as re
from collections import defaultdict
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer

from config import config
from tagolym import utils


def create_tag_mapping(tags):
    mappings = []
    for plb, clb in zip(config.PARTIAL_LABELS, config.COMPLETE_LABELS):
        similar_tags = set([t.lower() for tag in tags for t in tag if plb in t.lower()])
        mappings.append({tag: clb for tag in similar_tags})

    mapping = defaultdict(list)
    for mpg in mappings:
        for key, value in mpg.items():
            mapping[key].append(value)
    
    return mapping


def preprocess_tag(x, mapping):
    x = [tag.lower() for tag in x]       # lowercase all
    x = map(mapping.get, x)              # map tags
    x = filter(None, x)                  # remove None
    x = [t for tag in x for t in tag]    # flattened tags
    x = sorted(list(set(x)))             # drop duplicates and sort
    return x


def extract_features(equation_pattern, p):
    pattern = re.findall(equation_pattern, p)
    ptn_len = [len(ptn) for ptn in pattern]
    pattern = ["".join(ptn) for ptn in pattern]
    syntax = [" ".join(re.findall(r"\\(?:[^a-zA-Z]|[a-zA-Z]+[*=']?)", ptn)) for ptn in pattern]
    split = ["" if s is None else s for s in re.split(equation_pattern, p)]

    i = 0
    for ptn, length, cmd in zip(pattern, ptn_len, syntax):
        while "".join(split[i : i + length]) != ptn:
            i += 1
        split[i : i + length] = [cmd]

    return " ".join(split)


def preprocess_post(x, nocommand=False, stem=False):
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


def preprocess(df, nocommand, stem):
    mapping = create_tag_mapping(df["tags"])
    df["token"] = df["post_canonical"].apply(preprocess_post, args=(nocommand, stem))
    df["tags"] = df["tags"].apply(preprocess_tag, args=(mapping,))
    df = df[df["token"] != ""].reset_index(drop=True)
    return df


def binarize(tags):
    mlb = MultiLabelBinarizer()
    tags = mlb.fit_transform(tags)
    return tags, mlb


def split_data(X, y, train_size=0.7, random_state=None):
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