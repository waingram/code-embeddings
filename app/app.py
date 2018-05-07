import io
import tokenize
import csv
from javalang import tokenizer
from pathlib import Path

import gensim.models.doc2vec
from flask import Flask, render_template, request

app = Flask(__name__)


def tokenize_python_code(code):
    byte_str = io.BytesIO(code).read()  # assume code is a `BytesIO` object
    string_obj = byte_str.decode('utf-8')  # Convert to a unicode object

    tokens = list(tokenize.generate_tokens(io.StringIO(string_obj).readline))
    tokens = [token for t in tokens if t.type == tokenize.NAME or t.type == tokenize.OP for token in
              t.string.split(" ")]
    return tokens


def tokenize_java_code(code):
    byte_str = io.BytesIO(code).read()  # assume code is a `BytesIO` object
    string_obj = byte_str.decode('utf-8')  # Convert to a unicode object

    tokens = list(tokenizer.tokenize(string_obj))
    tokens = [token for t in tokens for token in t.value.split(" ")]
    return tokens


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    if request.method == 'GET':
        table_data = []
        return render_template('index.html', table_data=table_data, search_text='')
    else:
        lang = request.form['lang']
        search_text = request.form['search-text'].encode('utf-8')
        model_file = ""
        test_doc = ""
        model_map_file = ""
        if lang == "java":
            model_file = '../models/github-java-vectors.bin'
            test_doc = tokenize_java_code(search_text)
            model_map_file = '../models/java_doc_map.csv'
        elif lang == "python":
            model_file = '../models/github-python-vectors.bin'
            test_doc = tokenize_python_code(search_text)
            model_map_file = '../models/python_doc_map.csv'

        model = gensim.models.doc2vec.Doc2Vec.load(model_file)
        inferred_vector = model.infer_vector(test_doc, steps=200)
        sims = model.docvecs.most_similar([inferred_vector], topn=20)

        dict = {}
        with open(model_map_file, newline='') as csvfile:
            r = csv.reader(csvfile)
            for row in r:
                dict[int(row[0])] = row[1]

        table_data = []
        for j, t in enumerate(sims):
            f = Path('../', dict[t[0]])
            table_data.append([t[0], "{:2f}".format(t[1]), f.resolve().as_uri()])
            # code_str = ""
            # with f.open() as fin:
            #     code_str = fin.read()
            # table_data.append(code_str)

        return render_template('index.html', table_data=table_data, search_text=search_text)


if __name__ == '__main__':
    app.run()
