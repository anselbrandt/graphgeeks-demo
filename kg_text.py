import typing
import pathlib
import lancedb
import networkx as nx
import spacy
import pandas as pd
from icecream import ic
import gensim

from constants import SPACY_MODEL, STOP_WORDS
from utils import (
    init_nlp,
    TextChunk,
    Entity,
    parse_text,
    make_entity,
    extract_entity,
    extract_relations,
    connect_entities,
    defaultdict,
    run_textrank,
    abstract_overlay,
)
