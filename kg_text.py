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


def read_text(
    scrape_nlp: spacy.Language,
    url: str,
    chunk_list: typing.List[TextChunk],
    chunk_id: int,
) -> int:
    response: requests.Response = requests.get(
        url,
        headers=SCRAPE_HEADERS,
    )

    soup: BeautifulSoup = BeautifulSoup(
        response.text,
        features="lxml",
    )

    scrape_doc: spacy.tokens.doc.Doc = scrape_nlp(
        "\n".join([para.text.strip() for para in soup.find_all("p")])
    )

    chunk_id = make_chunk(
        scrape_doc,
        url,
        chunk_list,
        chunk_id,
    )

    return chunk_id


def construct_kg_text(
    files: typing.List[str],
    chunk_table: lancedb.table.LanceTable,
    sem_overlay: nx.Graph,
    w2v_file: pathlib.Path,
    *,
    debug: bool = True,
) -> None:
    """
    Construct a knowledge graph from unstructured data sources.
    """
    # define the global data structures which must be reset for each
    # run, not on each chunk iteration
    nlp: spacy.Language = init_nlp()
    known_lemma: typing.List[str] = []
    w2v_vectors: list = []

    # iterate through the URL list, scraping text and building chunks
    chunk_id: int = 0
    scrape_nlp: spacy.Language = spacy.load(SPACY_MODEL)

    for file in files:
        lex_graph: nx.Graph = nx.Graph()
        chunk_list: typing.List[TextChunk] = []

        chunk_id = read_text(
            scrape_nlp,
            file,
            chunk_list,
            chunk_id,
        )

        chunk_table.add(chunk_list)

        # parse each chunk to build a lexical graph per source URL
        for chunk in chunk_list:
            span_decoder: typing.Dict[tuple, Entity] = {}

            doc: spacy.tokens.doc.Doc = parse_text(
                nlp,
                known_lemma,
                lex_graph,
                chunk,
                debug=debug,
            )

            # keep track of sentence numbers per chunk, to use later
            # for entity co-occurrence links
            sent_map: typing.Dict[spacy.tokens.span.Span, int] = {}

            for sent_id, sent in enumerate(doc.sents):
                sent_map[sent] = sent_id

            # classify the recognized spans within this chunk as
            # potential entities

            # NB: if we'd run [_entity resolution_]
            # see: <https://neo4j.com/developer-blog/entity-resolved-knowledge-graphs/>
            # previously from _structured_ or _semi-structured_ data sources to
            # generate a "backbone" for the knowledge graph, then we could use
            # contextualized _surface forms_ perform _entity linking_ on the
            # entities extracted here from _unstructured_ data

            for span in doc.ents:
                make_entity(
                    span_decoder,
                    sent_map,
                    span,
                    chunk,
                    debug=debug,
                )

            for span in doc.noun_chunks:
                make_entity(
                    span_decoder,
                    sent_map,
                    span,
                    chunk,
                    debug=debug,
                )

            # overlay the recognized entity spans atop the base layer
            # constructed by _textgraph_ analysis of the `spaCy` parse trees
            for ent in span_decoder.values():
                if ent.key not in STOP_WORDS:
                    extract_entity(
                        known_lemma,
                        lex_graph,
                        ent,
                        debug=debug,
                    )

            # extract relations for co-occurring entity pairs
            extract_relations(
                known_lemma,
                lex_graph,
                span_decoder,
                sent_map,
                doc,
                chunk,
                debug=debug,
            )

            # connect entities which co-occur within the same sentence
            connect_entities(
                lex_graph,
                span_decoder,
            )

            # build the vector input for entity embeddings
            w2v_map: typing.Dict[int, typing.Set[str]] = defaultdict(set)

            for ent in span_decoder.values():
                if ent.node is not None:
                    w2v_map[ent.sent_id].add(ent.key)

            for sent_id, ents in w2v_map.items():
                vec: list = list(ents)
                vec.insert(0, str(sent_id))
                w2v_vectors.append(vec)

        # apply _textrank_ to the graph (in the url/doc iteration)
        # then report the top-ranked extracted entities
        df: pd.DataFrame = run_textrank(
            lex_graph,
        )

        ic(url, df.head(20))

        # abstract a semantic overlay from the lexical graph
        # and persist this in the resulting KG
        abstract_overlay(
            url,
            chunk_list,
            lex_graph,
            sem_overlay,
        )

        print("nodes", len(sem_overlay.nodes), "edges", len(sem_overlay.edges))

    # train the entity embedding model
    w2v_max: int = max([len(vec) - 1 for vec in w2v_vectors])

    w2v_model: gensim.models.Word2Vec = gensim.models.Word2Vec(
        w2v_vectors,
        min_count=2,
        window=w2v_max,
    )

    w2v_model.save(str(w2v_file))
