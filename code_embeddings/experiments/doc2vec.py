from code_embeddings.doc2vec_embedder import Doc2vecEmbedder
import numpy as np

code_dir = '../../data'
models_dir = '../../models'

doc2vec = Doc2vecEmbedder()
doc2vec.build_model(code_dir)
doc2vec.save_model(models_dir)

model = doc2vec.get_model()
docs = doc2vec.get_docs()

doc_id = np.random.randint(model.docvecs.count)  # Pick random doc; re-run cell for more examples
doc_id = 'sieve-of-eratosthenes-6.java'

doc = [doc for doc in docs if doc.tags[0] == 'sieve-of-eratosthenes-6.java'][0]
inferred_docvec = model.infer_vector(doc.words)

print('\n%s' % (doc.tags[0]))
for vec in model.docvecs.most_similar([inferred_docvec], topn=6):
    print('\t%s' % str(vec))
