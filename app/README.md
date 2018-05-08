## Code Similarity Search Engine

As a proof of concept, we developed a search engine that makes use of our doc2vec models for searching a corpus of
source code. The search engine takes a text block as input. The user may enter a Java or Python code fragment of
arbitrary length. When submitted, the application will use the corresponding model (Java or Python) to infer a vector
for the given code fragment. Then, the inferred vector is used to find the top-n most-similar vectors known from the
training by calculating the cosine-similarity. The results are displayed to the user.

### Running the code

This is a simple flask app. Create a virtual environment, activate it, and install the dependencies listed at the
project root. Then run:

    $ python app.py