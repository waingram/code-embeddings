from code_embeddings.doc2vec_embedder import Doc2vecEmbedder
import numpy as np

code_dir = '../../data'
models_dir = '../../models'

doc2vec = Doc2vecEmbedder()
doc2vec.build_model(code_dir)
doc2vec.train()
doc2vec.save_model(models_dir)

model = doc2vec.get_model()
docs = doc2vec.get_docs()

doc_id = np.random.randint(model.docvecs.count)  # Pick random doc; re-run cell for more examples
doc_id = [i for i, x in enumerate(docs) if x[1][0] == 'Sieve-of-Eratosthenes'][4]

inferred_docvec = model.infer_vector(docs[doc_id].words)
print('\n%s:' % (docs[doc_id].tags[0]))
for vec in model.docvecs.most_similar([inferred_docvec], topn=model.docvecs.count):
    print('%s' % str(vec))