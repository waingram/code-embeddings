from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import gensim.models.doc2vec
from code_embeddings.utils import tokenize
from pathlib import Path
import multiprocessing
from random import shuffle


class Doc2vecEmbedder():
    """
        Doc2Vec Parameters
        ----------
        documents : iterable of iterables
            The `documents` iterable can be simply a list of TaggedDocument elements, but for larger corpora,
            consider an iterable that streams the documents directly from disk/network.
            If you don't supply `documents`, the model is left uninitialized -- use if
            you plan to initialize it in some other way.

        dm : int {1,0}
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.

        size : int
            Dimensionality of the feature vectors.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        iter : int
            Number of iterations (epochs) over the corpus.
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        dm_mean : int {1,0}
            If 0 , use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm_concat : int {1,0}
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int
            Expected constant number of document tags per document, when using
            dm_concat mode; default is 1.
        dbow_words : int {1,0}
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

    """

    _cores = multiprocessing.cpu_count()
    _passes = 20

    _dm = 1  # training algorithm: 1 = PV-DM, 0 = PV-DBOW
    _vector_size = 40
    _window = 5
    _alpha = 0.025
    _min_alpha = 0.001
    _min_count = 2
    _max_vocab_size = None
    _sample = 1e-5
    _workers = _cores
    _epochs = 500
    _hs = 0
    _negative = 0
    _dm_mean = 0
    _dm_concat = 0
    _dm_tag_count = 2
    _dbow_words = 1

    _alpha_delta = (_alpha - _min_alpha) / _passes

    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    def __init__(self):
        self._code_directory = None
        self._models_directory = None
        self._tagged_docs = None
        # TODO: make these parameters
        self._model = Doc2Vec(dm=self._dm,
                              vector_size=self._vector_size,
                              window=self._window,
                              alpha=self._alpha,
                              min_alpha=self._min_alpha,
                              min_count=self._min_count,
                              max_vocab_size=self._max_vocab_size,
                              sample=self._sample,
                              workers=self._workers,
                              epochs=self._epochs,
                              hs=self._hs,
                              negative=self._negative,
                              dm_mean=self._dm_mean,
                              dm_concat=self._dm_concat,
                              dm_tag_count=self._dm_tag_count,
                              dbow_words=self._dbow_words)

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
        for epoch in range(self._passes):
            doc_list = self._tagged_docs
            shuffle(doc_list)  # Shuffling gets best results
            if epoch % 2 == 0:
                print('Now training epoch {}'.format(epoch))
            self._model.train(self._tagged_docs, total_examples=self._model.corpus_count, epochs=self._model.epochs)
            self._model.alpha -= self._min_alpha  # decrease the learning rate
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
