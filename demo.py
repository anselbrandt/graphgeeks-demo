from collections import defaultdict
from dataclasses import dataclass
import enum
import itertools
import json
import logging
import math
import os
import pathlib
import sys
import traceback
import tracemalloc
import typing
import unicodedata
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from bs4 import BeautifulSoup
from gliner_spacy.pipeline import GlinerSpacy
from icecream import ic
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pyinstrument import Profiler
import gensim
import glirel
import lancedb
import networkx as nx
import numpy as np
import pandas as pd
import pyvis
import requests
import spacy
import transformers

from constants import (
    CHUNK_SIZE,
    EMBED_FCN,
    GLINER_MODEL,
    LANCEDB_URI,
    NER_LABELS,
    RE_LABELS,
    SCRAPE_HEADERS,
    SPACY_MODEL,
    STOP_WORDS,
    TR_ALPHA,
    TR_LOOKBACK,
)

from utils import (
    TextChunk,
    Entity,
    uni_scrubber,
    make_chunk,
    scrape_html,
    init_nlp,
    parse_text,
    make_entity,
    extract_entity,
    extract_relations,
    calc_quantile_bins,
    stripe_column,
    root_mean_square,
    connect_entities,
    run_textrank,
    abstract_overlay,
    gen_pyvis,
    construct_kg,
)

if __name__ == "__main__":
    # start the stochastic call trace profiler and memory profiler
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # define the global data structures
    url_list: typing.List[str] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    ]

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        "chunk",
        schema=TextChunk,
        mode="overwrite",
    )

    sem_overlay: nx.Graph = nx.Graph()

    try:
        construct_kg(
            url_list,
            chunk_table,
            sem_overlay,
            pathlib.Path("data/entity.w2v"),
            debug=False,  # True
        )

        # serialize the resulting KG
        with pathlib.Path("data/kg.json").open("w", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(sem_overlay),
                    indent=2,
                    sort_keys=True,
                )
            )

        # generate HTML for an interactive visualization of a graph
        gen_pyvis(
            sem_overlay,
            "kg.html",
            num_docs=len(url_list),
        )
    except Exception as ex:
        ic(ex)
        traceback.print_exc()

    # stop the profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report the memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")
