import numpy as np
import scipy
import ssqueezepy as ssq

class Resample:
  def __init__(self, num):
    self._num = num

  def __call__(self, x):
    return scipy.signal.resample(x, self._num)

class FSST:
  def __init__(self, fs, flipud=False, window=None):
    self.flipud = flipud
    self.fs = fs
    self.window = window

  def __call__(self, x):
    Tsx, Sx, f, *_ = ssq.ssq_stft(
        x,
        flipud=self.flipud,
        fs=self.fs,
        window=self.window,
    )

    return Tsx, Sx, np.ascontiguousarray(f)