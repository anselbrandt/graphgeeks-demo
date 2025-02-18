import json
from pathlib import Path
import traceback
import tracemalloc
import typing
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from icecream import ic
from pyinstrument import Profiler
import lancedb
import networkx as nx

from constants import (
    LANCEDB_URI,
)

from utils import (
    TextChunk,
    gen_pyvis,
)

from kg_text import construct_kg_text

from transcript_utils import srt_to_lines, srt_to_text

if __name__ == "__main__":
    # start the stochastic call trace profiler and memory profiler
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    files = [file for file in Path("test").iterdir()]

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        "chunk",
        schema=TextChunk,
        mode="overwrite",
    )

    sem_overlay: nx.Graph = nx.Graph()

    try:
        construct_kg_text(
            files,
            chunk_table,
            sem_overlay,
            Path("data/entity.w2v"),
            debug=False,  # True
        )

        # serialize the resulting KG
        with Path("data/kg.json").open("w", encoding="utf-8") as fp:
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
            num_docs=len(files),
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
