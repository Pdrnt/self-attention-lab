import numpy as np


class ScaledDotProductAttention:

    def forward(self, Q, K, V):
        # 1️⃣ Produto escalar QK^T
        scores = np.matmul(Q, K.T)

        # 2️⃣ Scaling factor
        d_k = K.shape[-1]
        scaled_scores = scores / np.sqrt(d_k)

        return scaled_scores
