import tracemalloc
import spacy
import sys
from multiprocessing import current_process

words = ['dog', 'cat', 'fox', 'chicken', 'cow', 'mouse']


def work_with_shared_memory(work, model_path, shm_name, rows, cols, dtype):
    tracemalloc.start()
    shape = (rows, cols)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current Process: {current_process().name=} memory usage {current / 1e6}MB; Peak: {peak / 1e6}MB")
    shared = {'vectors': {'name': shm_name, 'shape': shape, 'dtype': dtype}}
    nlp = spacy.load(model_path, shared=shared)
    sims = []
    for pair in work:
        # data is sent as numpy.str...
        w1 = nlp(str(pair[0]))
        w2 = nlp(str(pair[1]))
        sim = w1.similarity(w2)
        sims.append(sim)
    sys.stdout.flush()
    tracemalloc.stop()
    nlp.vocab.vectors.close_shared_memory()
    return sims


def get_nlp_similarity(w1, w2, model):
    sim = None
    if w1 and w2:
        w1v = model(w1)
        w2v = model(w2)
        sim = w1v.similarity(w2v)
    return sim

