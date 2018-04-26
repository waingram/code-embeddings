import io
import tokenize
from pathlib import Path

import gensim.models.doc2vec
from flask import Flask, render_template, request

app = Flask(__name__)
model_file = '../models/github-python-vectors.bin'
model = gensim.models.doc2vec.Doc2Vec.load(model_file)


def tokenize_python_code(code):
    byte_str = io.BytesIO(code).read()  # assume code is a `BytesIO` object
    string_obj = byte_str.decode('utf-8')  # Convert to a unicode object

    tokens = list(tokenize.generate_tokens(io.StringIO(string_obj).readline))
    tokens = [token for t in tokens if t.type == tokenize.NAME or t.type == tokenize.OP for token in
              t.string.split(" ")]
    return tokens


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'GET':
        table_data = []
        return render_template('index.html', table_data=table_data, search_text='')
    else:
        search_text = request.form['search-text'].encode('utf-8')
        test_doc = tokenize_python_code(search_text)
        inferred_vector = model.infer_vector(test_doc, steps=200)
        sims = model.docvecs.most_similar([inferred_vector], topn=5)
        table_data = []
        for j, t in enumerate(sims):
            table_data.append([t[0], "{:2f}".format(t[1])])

        return render_template('index.html', table_data=table_data, search_text=search_text)


if __name__ == '__main__':
    app.run()
