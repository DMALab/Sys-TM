from ctypes import *

lib = cdll.LoadLibrary('./libliblda.so')

class LDA(object):
    def __init__(self, n_topics, alpha=None, beta=None):
	if alpha is None:
	    alpha = 50.0 / n_topics
	if beta is None:
	    beta = 0.01
	self.obj = lib.Lda_new(c_int(n_topics), c_double(alpha), c_double(beta))

    def train(self, corpus, max_iter=100):
	lib.Lda_train(self.obj, corpus.obj, c_int(max_iter))

class SLDA(object):
    def __init__(self, n_topics, alpha=None, beta=None, sigma2=None):
        if alpha is None:
	    alpha = 50.0 / n_topics
	if beta is None:
	    beta = 0.01
	if sigma2 is None:
	    sigma2 = 1.0
	self.obj = lib.slda_new(c_int(n_topics), c_double(alpha), c_double(beta), c_double(sigma2))
    def train(self, corpus, max_iter=100):
	lib.SLda_train(self.obj, corpus.obj, c_int(max_iter))

class CTM(object):
    def __init__(self, n_topics, mu=None, sigma=None, beta=None):
        if alpha is None:
	    alpha = 50.0 / n_topics
	if beta is None:
	    beta = 0.01
	if sigma is None:
	    sigma = 1.0
	if mu is None:
	    mu = None
	self.obj = lib.Ctm_new(c_int(n_topics), c_double(mu), c_double(sigma), c_double(beta)) 
    def train(self, corpus, max_iter=100):
	lib.Ctm_train(self.obj, corpus.obj, c_int(max_iter))

class TOT(object):
    def __init__(self, n_topics, alpha=None, beta=None):
	if alpha is None:
	    alpha = 50.0 / n_topics
	if beta is None:
	    beta = 0.01
	self.obj = lib.tot_new(c_int(n_topics), c_double(alpha), c_double(beta))

    def train(self, corpus, max_iter=100):
	lib.Tot_train(self.obj, corpus.obj, c_int(max_iter))


class Corpus(object):
    def __init__(self, path):
	self.obj = lib.Corpus_read_from_file(c_char_p(path))
