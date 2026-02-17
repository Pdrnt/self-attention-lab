import numpy as np
from attention import ScaledDotProductAttention


def main():
    # Exemplo simples
    Q = np.array([[1, 0, 1],
                  [0, 1, 0]])

    K = np.array([[1, 0, 1],
                  [0, 1, 0]])

    V = np.array([[1, 2],
                  [3, 4]])

    attention = ScaledDotProductAttention()
    output, weights = attention.forward(Q, K, V)

    print("Attention Weights:")
    print(weights)

    print("\nOutput:")
    print(output)


if __name__ == "__main__":
    main()
