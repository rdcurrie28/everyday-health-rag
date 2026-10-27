"""Load the search index, then start Streamlit in the same process.

Cloud Run routes traffic only once Streamlit is listening, so after a cold start the index is
already in memory and the first visitor does not wait for it.
"""

import sys

from streamlit.web import cli

from healthrag.config import get_settings
from healthrag.retrieval import IndexNotReady, shared_retriever


def main() -> None:
    try:
        shared_retriever(get_settings())
    except IndexNotReady:
        pass  # app.py shows the error to visitors
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
