# Scaled Dot-Product Attention

Implementação do mecanismo Scaled Dot-Product Attention descrito no paper:

Attention Is All You Need (Vaswani et al., 2017)

## Descrição

Este projeto implementa a fórmula:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

A implementação foi feita utilizando apenas NumPy, sem bibliotecas de alto nível como PyTorch ou TensorFlow.

---

## Explicação do Scaling Factor (√d_k)

Após o cálculo do produto escalar QK^T, o resultado é dividido pela raiz quadrada da dimensão das chaves (d_k).

scaled_scores = scores / sqrt(d_k)

Essa normalização é necessária para evitar que os valores do produto escalar se tornem muito grandes, o que poderia causar saturação do Softmax e instabilidade numérica.

---

## Estrutura do Projeto

- attention.py → Implementação da classe ScaledDotProductAttention
- test_attention.py → Script de validação com exemplo numérico
- requirements.txt → Dependências do projeto

---

## Como Executar

1. Criar e ativar o ambiente virtual:

python3 -m venv venv
source venv/bin/activate

2. Instalar dependências:

pip install -r requirements.txt

3. Executar o teste:

python test_attention.py


---

## Exemplo de Entrada

Q = [[1, 0, 1],
     [0, 1, 0]]

K = [[1, 0, 1],
     [0, 1, 0]]

V = [[1, 2],
     [3, 4]]

---

## Saída Esperada

- Matriz de pesos de atenção (cada linha soma aproximadamente 1)
- Matriz final resultante da multiplicação pelos valores (V)