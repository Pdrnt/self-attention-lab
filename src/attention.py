import numpy as np


class ScaledDotProductAttention:

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, Q, K, V):
        # 1️⃣ Produto escalar QK^T
        scores = np.matmul(Q, K.T)

        # 2️⃣ Scaling factor
        d_k = K.shape[-1]
        scaled_scores = scores / np.sqrt(d_k)

        # 3️⃣ Softmax
        attention_weights = self.softmax(scaled_scores)

        # 4️⃣ Multiplicação pelos Values
        output = np.matmul(attention_weights, V)

        return output, attention_weights
