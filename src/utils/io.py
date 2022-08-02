import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import joblib
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
import os
import sys

class ProgressParallel(joblib.Parallel):
    ## Allowing for progress bar to be shown.
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

# deletes all contents of dir and recreates it.
def create_dir(path):
  # if os.path.exists(path):
  #   print('Path ', path, ' already exists. Please delete and restart your job.')
  #   sys.exit(1)
  os.makedirs(path, exist_ok=True)

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
  """
  Extends dgllife pmap function.

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = ProgressParallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(d, **kwargs) for i, d in enumerate(data)
  )

  return results