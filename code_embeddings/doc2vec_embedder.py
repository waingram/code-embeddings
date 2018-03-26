from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import gensim.models.doc2vec
from code_embeddings.utils import tokenize
from pathlib import Path
import multiprocessing
from random import shuffle


class Doc2vecEmbedder():
    cores = multiprocessing.cpu_count()
    passes = 20
    alpha = 0.025
    min_alpha = 0.001
    training_algorithm = 2
    epochs = 500
    vector_size = 50
    window = 5
    min_count = 10
    alpha = 0.05
    negative = 0
    alpha_delta = (alpha - min_alpha) / passes
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    def __init__(self):
        self._code_directory = None
        self._models_directory = None
        self._tagged_docs = None
        # TODO: make these parameters
        self._model = Doc2Vec(dm=self.training_algorithm,
                              dbow_words=1,
                              sample=1e-5,
                              alpha=self.alpha,
                              min_alpha=self.min_alpha,
                              vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              negative=self.negative,
                              epochs=self.epochs,
                              workers=self.cores)

    def _process_files(self):
        for programming_language in self._code_directory.glob('./Java'):
            if not programming_language.is_dir():
                continue
            for programming_task in programming_language.glob('./*'):
                if not programming_task.is_dir():
                    continue
                for code_fragment in programming_task.glob('./*'):
                    # print('Processing "{}"'.format(code_fragment))
                    with code_fragment.open() as f:
                        tokens = tokenize(f.read())
                    yield TaggedDocument(tokens, [programming_task.name, programming_language.name])

    def build_model(self, code_directory):
        self._code_directory = Path(code_directory)
        self._tagged_docs = list(self._process_files())
        print('Doc2Vec: building the model')
        self._model.build_vocab(self._tagged_docs)

    def train(self):
        for epoch in range(self.passes):
            doc_list = self._tagged_docs
            shuffle(doc_list)  # Shuffling gets best results
            if epoch % 2 == 0:
                print('Now training epoch {}'.format(epoch))
            self._model.train(self._tagged_docs, total_examples=self._model.corpus_count, epochs=self._model.epochs)
            self._model.alpha -= self.min_alpha  # decrease the learning rate
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
