from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from code_embeddings.utils import tokenize
from pathlib import Path


class Doc2vecEmbedder():

    def __init__(self):
        self._code_directory = None
        self._models_directory = None
        self._tagged_docs = None
        self._model = None

    def _process_files(self):
        for programming_task in self._code_directory.glob('./*/*'):
            if not programming_task.is_dir():
                continue
            for code_fragment in programming_task.glob('./*'):
                # print('Processing "{}"'.format(code_fragment))
                with code_fragment.open() as f:
                    tokens = tokenize(f.read())
                yield TaggedDocument(tokens, [programming_task.name])

    def build_model(self, code_directory):
        self._code_directory = Path(code_directory)
        self._tagged_docs = list(self._process_files())
        print('Doc2Vec: building the model')
        self._model = Doc2Vec(self._tagged_docs, vector_size=100, window=8, min_count=2,
                              workers=7)  # TODO: make these parameters

    def train(self):
        for epoch in range(10):
            if epoch % 2 == 0:
                print('Now training epoch {}'.format(epoch))
            self._model.train(self._tagged_docs, total_examples=self._model.corpus_count, epochs=self._model.epochs)
            self._model.alpha -= 0.002  # decrease the learning rate
            self._model.min_alpha = self._model.alpha  # fix the learning rate, no decay

    def save_model(self, models_directory):
        self._models_directory = Path(models_directory)
        file_path = self._models_directory / 'model.doc2vec.gz'
        self._model.save(str(file_path))

    def load_model(self, models_directory):
        self._models_directory = Path(models_directory)
        file_path = self._models_directory / 'model.doc2vec.gz'
        self._model = Doc2Vec.load(str(file_path))

    def get_model(self) -> Doc2Vec:
        return self._model

    def get_docs(self) -> TaggedDocument:
        return self._tagged_docs
