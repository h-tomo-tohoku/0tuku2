import sys

sys.path.append("..")
import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from skip_gram import SkipGram
from common.util import create_contexts_target
from dataset import ptb

window_size = 5
hidden_size = 10
batch_size = 100
max_epoch = 10

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vec
params = {}
params["word_vec"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = "skip_gram_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
