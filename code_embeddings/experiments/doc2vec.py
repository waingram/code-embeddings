from code_embeddings.doc2vec_embedder import Doc2vecEmbedder

code_dir = '../../data'
models_dir = '../../models'

doc2vec = Doc2vecEmbedder()
doc2vec.build_model(code_dir)
doc2vec.train()
doc2vec.save_model(models_dir)

model = doc2vec.get_model()
docs = doc2vec.get_docs()
print(docs[13])
# model.docvecs.most_similar(docs[13])





