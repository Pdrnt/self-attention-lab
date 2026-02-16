import numpy as np


class ScaledDotProductAttention:

    def forward(self, Q, K, V):
        # 1️⃣ Produto escalar entre Q e K^T
        scores = np.matmul(Q, K.T)

        return scores
