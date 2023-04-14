import sys

sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram:
    def __init__(
        self,
        vocab_size,
        hidden_size,
    ) -> None:
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        # レイヤの生成
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        layers = [self.in_layer, self.out_layer, self.loss_layer0, self.loss_layer1]
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を固定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l0 = self.loss_layer0.forward(s, contexts[:, 0])
        l1 = self.loss_layer1.forward(s, contexts[:, 1])
        loss = l0 + l1
        return loss

    def backward(self, dout=1):
        d0 = self.loss_layer0.backward(dout)
        d1 = self.loss_layer1.backward(dout)
        ds = d0 + d1
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
