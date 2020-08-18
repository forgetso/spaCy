from multiprocessing import cpu_count, set_start_method, Pool
import numpy as np
from shared_memory_worker import work_with_shared_memory
import spacy
import random
import time

set_start_method('fork', force=True)
words = ['dog', 'cat', 'fox', 'chicken', 'cow', 'mouse']


# split a list into evenly sized chunks
def chunks(lst, n):
    if n < 1:
        n = 1
    result = [x for x in np.array_split(lst, n) if len(x)]
    return result


# get a pool of workers to process the data
def job_pool(data, job_number, job_to_do, **kwargs):
    slices = chunks(data, job_number)

    if len(slices) < job_number:
        job_number = len(slices)

    pool = Pool(processes=job_number)

    # add the extra arguments needed by the job_to_do
    if kwargs is not None:
        arg = [(job_to_do, s, kwargs) for s in slices]
    else:
        arg = [(job_to_do, s) for s in slices]
    jobs = pool.map(worker_wrapper, arg)

    return jobs


def worker_wrapper(arg):
    job_to_do, data, kwargs = arg
    return job_to_do(data, **kwargs)


def main():
    # Load the spacy model - a shared memory for vectors will be created
    model_path = '/home/chris/dev/uniqueness/data/crawl'
    nlp = spacy.load(model_path)
    shape, dtype = nlp.vocab.vectors.data.base.shape, nlp.vocab.vectors.data.base.dtype
    shm = nlp.vocab.vectors.shm
    process_count = int(cpu_count() / 2)
    work = []
    for _ in range(0, process_count * 10000):
        work.append((random.choice(words), random.choice(words)))
    start = time.time()
    processes = job_pool(data=work,
                         model_path=model_path,
                         job_number=process_count,
                         job_to_do=work_with_shared_memory,
                         shm_name=shm.name,
                         rows=shape[0],
                         cols=shape[1],
                         dtype=dtype)
    end = time.time()
    result = np.concatenate([np.array(x) for x in processes])
    print('got {} similarities from {} jobs'.format(len(result), process_count))
    print('time taken {}s'.format(start - end))
    nlp.vocab.vectors.close_shared_memory()
    nlp.vocab.vectors.unlink_shared_memory()


if __name__ == "__main__":
    main()
