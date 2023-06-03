import mlflow
from pathlib import Path
from nltk.corpus import stopwords

# set seed for reproducibility
SEED = 42

# directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "stores")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# experiment
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# preprocessing
STOPWORDS = stopwords.words("english")
COMMANDS = ["prove", "let", "find", "show", "given"]

# regex pattern
EQUATION_PATTERN = (
    r"(?<!\\)"    # negative look-behind to make sure start is not escaped
    r"(?:"        # start non-capture group for all possible match starts
        # group 1, match dollar signs only
        # single or double dollar sign enforced by look-arounds
        r"((?<!\$)\${1,2}(?!\$))|"
        # group 2, match escaped parenthesis
        r"(\\\()|"
        # group 3, match escaped bracket
        r"(\\\[)|"
        # group 4, match begin equation
        r"(\\begin)"
    r")"
    # if group 1 was start
    r"(?(1)"
        # non greedy match everything in between
        # group 1 matches do not support recursion
        r"(.*?)(?<!\\)"
        # match ending double or single dollar signs
        r"(?<!\$)\1(?!\$)|"
    # else
    r"(?:"
        # greedily and recursively match everything in between
        # groups 2, 3 and 4 support recursion
        r"(.*(?R)?.*)(?<!\\)"
        r"(?:"
            # if group 2 was start, escaped parenthesis is end
            r"(?(2)\\\)|"
            # if group 3 was start, escaped bracket is end
            r"(?(3)\\\]|"
            # else group 4 was start, match end equation
            r"\\end"
        r")"
    r"))))"
)

# regex pattern
ASYMPTOTE_PATTERN = (
    r"(?<!\\)"    # negative look-behind to make sure start is not escaped
    r"(?:"        # start non-capture group for all possible match starts
        # group 1, match begin asymptote
        r"(\[asy\])"
    r")"
    # if group 1 was start
    r"(?(1)"
        # non greedy match everything in between
        # group 1 matches do not support recursion
        r"(.*?)(?<!\\)"
        # match ending asymptote
        r"\[/asy\]"
    r")"
)

# labels generation
PARTIAL_LABELS = [
    "algebra", "geometr", "number theor", "combinator",    # big topics
    "inequalit", "function", "polynomial",                 # algebra subtopics
    "circle", "trigonometr",                               # geometry subtopics
    "modul",                                               # number theory subtopics
]
COMPLETE_LABELS = [
    "algebra", "geometry", "number theory", "combinatorics",    # big topics
    "inequality", "function", "polynomial",                     # algebra subtopics
    "circle", "trigonometry",                                   # geometry subtopics
    "modular arithmetic",                                       # number theory subtopics
]