# Simple Wikipedia Search Engine
The goal of this project is to create a simple search engine for a subset of wikipedia pages by implementing basic information retrieval algorithms, like the tf-idf model and word2vec & doc2vec embeddings.

## How to use it
### Crawling a Wikipedia category
The first step in Information Retrieval is to crawl a Wikipedia category through the links presents in each page. Crawling depth can be configured in the script `crawl.py`.
```
python3 crawl.py
```

### Downloading pages content in XML
Then, we need to fetch the XML pages to the `/dws` directory with the [Wikipedia API](https://en.wikipedia.org/wiki/Special:Export)
```
./dw.sh wiki.lst
```

### Parsing XML pages : links extraction, tokenization, stemming
We can create dictionnaries containing tokens and links informations for each page on a reverse sparse index format by running:
```
python3 parsexml.py dws/*
```
It will compute every data structure needed for tf-idf and word2vec embeddings and store them as Pickle files so that they can be used later by the other scripts.

### Computing the PageRank vector from the links information
From the links informations previously retrieved we can compute a page ranking and store it as a Pickle file:
```
python3 pageRank.py
```

### Querying the Wikipedia page set
Now we have everything we need to do a search on this set of documents. Running the `search.py` script should give the most relevant documents to our query !

```
python3 search.py "<query>" <number of results to show>
```

Example:
```
python3 search.py "evolution bacteria" 5
```

## Understand each script step by step
The source codes are commented enough to understand each step of the process. Moreover, a `DEBUG` option can be set to `True` for `parsexml.py`, `pageRank.py` and `search.py` to generate intermediate JSON dumps and be able to visualize each calculation step.

> Note: I surely could have optimized some instructions in order to be more efficient, but the goal of this work being above all to be educational, I chose to focus on readability and understanding rather than performance.

This work was done as part of the Semantic Web and Information Retrieval course given in the last year of the Information Systems Engineering specialization at Grenoble INP - Ensimag, UGA.
