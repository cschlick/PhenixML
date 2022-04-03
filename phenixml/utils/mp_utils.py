from multiprocessing import Pool
from contextlib import closing
import tqdm

def pool_with_progress(worker,work,nproc=16,disable_progress=False,**kwargs):
  results = []
  with closing(Pool(processes=nproc)) as pool:
    for result in tqdm.tqdm(pool.imap(worker, work), total=len(work),disable=disable_progress):
        results.append(result)

  del pool
  
  return results