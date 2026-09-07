# HNSW from scratch

Implementação didática de HNSW do zero, com benchmark contra busca vetorial exata.

## Foco

A implementação de `src/hnsw.py` fica propositalmente em aberto. O restante do repositório já fornece a infraestrutura para responder à pergunta da Spike: **a partir de qual volume de embeddings HNSW passa a compensar em relação à busca exata?**

## Contrato esperado do HNSW

```python
index = HNSWIndex(
    m=16,
    ef_construction=200,
    ef_search=50,
)

index.build(vectors)
results = index.search(query, k=10)
```

`search()` deve retornar os índices dos vetores encontrados, ordenados do mais próximo para o menos próximo.

## Benchmark

Instale as dependências:

```bash
pip install -r requirements.txt
```

Quando o HNSW estiver implementado:

```bash
python benchmarks/run_benchmark.py
```

Por padrão são testados vetores de 1536 dimensões com bases de:

```text
1.000
5.000
10.000
25.000
50.000
```

Para aumentar a escala:

```bash
python benchmarks/run_benchmark.py --sizes 10000 25000 50000 100000
```

O benchmark mede:

- p50/p95 e média de latência;
- tempo de construção;
- variação aproximada de RSS;
- Recall@K contra busca exata;
- speedup de p95.

Os resultados são escritos em `benchmarks/results/results.csv`.

## Gráficos

Depois do benchmark:

```bash
python benchmarks/plot_results.py
```

Isso gera os gráficos de latência p95 e Recall@K em `benchmarks/results/`.

## Estrutura

```text
src/
├── brute_force.py   # baseline exata / ground truth
└── hnsw.py          # implementação manual

benchmarks/
├── metrics.py
├── run_benchmark.py
└── plot_results.py
```
