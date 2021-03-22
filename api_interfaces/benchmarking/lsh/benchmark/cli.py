from pprint import pprint
from benchmark_lsh import get_response, benchmark
from typer import Argument, Option, Typer

cli = Typer(help=__doc__, no_args_is_help=True)


@cli.command()
def benchmark(
    n_classifiers: int = Option(
        ..., help="The number of classifiers, ie the number of chunks the feature vectors has been split into, or the number of tokens in each LSH hash"
    ),
    n_clusters: int = Option(
        ..., help="The number of clusters found by each classifier"
    ),
    sample_size: int = Option(
        default=250,
        help="The number of times the API will be hit to determine an average response time"
    )
):
    print("Example response:")
    pprint(get_response(n_classifiers, n_clusters))
    print()
    benchmark(n_classifiers, n_clusters, sample_size)


if __name__ == "__main__":
    cli()
