import hashlib
import re
import struct
from itertools import combinations, tee
from typing import Iterable, List, Tuple
from logging import Logger

import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.integrate import quad as integrate

SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


def ngrams(sequence: List[str], n: int) -> Iterable:
    """
    Code taken from NLTK, without padding.

    Parameters
    ----------
    sequence : list
        The sequence of items to be converted into n-grams.
    n : int
        The order of the n-grams to be extracted.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def generate_hash_values(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
):
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    hv = np.array([sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64)  # noqa: E501
    a, b = permutations  # [1, num_perm], [1, num_perm]
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    return [
        (table_idx, bytes(hashvalues[start:end].byteswap().data), idx)
        for table_idx, (start, end) in enumerate(hashranges)
    ]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


conf = SparkConf()
conf.set("spark.app.name", "MinHashLSH")
conf.set("spark.debug.maxToStringFields", "100")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore

threshold = 0.7
ngram_size = 5
num_perm = 256
B, R = optimal_param(threshold, num_perm)
HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
PERMUTATIONS = np.array(
    [
        (
            RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
            RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
        )
        for _ in range(num_perm)
    ],
    dtype=np.uint64,
).T  # [2, num_perm]

table = "huggingface-science-codeparrot.the_stack_java.java"

df = spark.read.format("bigquery").option("table", table).load().limit(1_000_000)
df = df.withColumn("__id__", F.monotonically_increasing_id())
records = df.select("__id__", "content").rdd
records = records.repartition(num_perm)

edges = (
    records.flatMap(
        lambda x: generate_hash_values(
            content=x[1],
            idx=x[0],
            num_perm=num_perm,
            ngram_size=ngram_size,
            hashranges=HASH_RANGES,
            permutations=PERMUTATIONS,
        )
    )
    .groupBy(lambda x: (x[0], x[1]))  # (table_idx, hashvalue)
    .filter(lambda x: len(list(x[1])) > 1)
    .flatMap(lambda x: list(combinations([i[2] for i in x[1]], 2)))  # (idx1, idx2)
    .distinct()
).cache()

# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


a = edges
while True:
    b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
    a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
    changes = a.subtract(b).union(b.subtract(a)).collect()
    if len(changes) == 0:
        break

results = a.collect()
components = spark.createDataFrame(results, schema=["__id__", "component"]).sort(["component", "__id__"])
components.show()

df = df.join(components, on="__id__", how="left")
df = df.filter(F.col("component").isNull()).drop("__id__", "component").cache()
df.write.json("gs://chenghao-data/dataproc_output/deduplicated", mode="overwrite")

# export CLUSTER_NAME=chenghao-temp
# gcloud dataproc jobs submit pyspark --cluster ${CLUSTER_NAME} --region us-central1 \
# --jars gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
# --driver-log-levels root=WARN \
# --properties="spark.executor.memory"="20g",\
# "spark.driver.memory"="32g",\
# "spark.executor.cores"="8" \
# spark.py
