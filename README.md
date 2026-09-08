# HNSW from scratch

Implementação didática de HNSW do zero, com benchmark contra busca vetorial exata.

## Objetivo da Spike

O benchmark foi ajustado para se aproximar do fluxo atual de **document retrieval** da aplicação analisada, sem tentar reproduzir toda a stack de banco, autorização e ingestão.

O foco experimental é responder:

> **A partir de qual volume efetivo de embeddings pesquisáveis HNSW passa a reduzir a latência em relação à busca exata, mantendo Recall@3 aceitável?**

## Características reproduzidas

O fluxo atual de recuperação trabalha, no cenário analisado, com:

- embeddings de **1536 dimensões**;
- comparação por **cosine**;
- recuperação de **Top-K = 3** no chat;
- vetores representando chunks/documentos;
- filtros de perfil, domínio/namespace e acesso antes da resposta final.

O benchmark usa `n` como o número de embeddings **efetivamente pesquisáveis**. Em outras palavras, se o banco possui 100.000 chunks, mas os filtros deixam 18.000 candidatos para uma consulta, o valor mais relevante para este benchmark é 18.000.

A aplicação transforma a distância em score com uma função monotônica. Para o ranking dos vizinhos, ordenar por maior cosine similarity é equivalente a ordenar por menor cosine distance e também preserva a ordem desse score. Por isso, o benchmark compara diretamente os vizinhos cosine sem reproduzir a transformação de score.

## Dataset sintético

O benchmark possui dois modos.

### `clustered` — padrão

Gera vários grupos semânticos dentro do mesmo universo de documentos. Os clusters representam assuntos/intents/chunks semanticamente próximos, **não domínios diferentes**.

Exemplo conceitual:

```text
                dp-rh

benefícios          ponto             treinamento
● ● ● ●             ● ● ● ●           ● ● ●
VR                   Ahgora            cursos
Flash                jornada           eficácia
VT                   marcação          capacitação
```

O HNSW não conhece esses rótulos. Ele recebe apenas os vetores. Os clusters existem somente para tornar a distribuição sintética menos artificial do que vetores totalmente independentes.

### `random` — controle

Mantém o cenário anterior, com vetores unitários aleatórios e independentes. É útil como controle computacional, mas é menos representativo de uma base semântica real.

## Contrato do HNSW

```python
index = HNSWIndex(
    m=16,
    ef_construction=200,
    ef_search=50,
)

index.build(vectors)
results = index.search(query, k=3)
```

`search()` retorna os índices dos vetores encontrados, ordenados do mais próximo para o menos próximo.

## Benchmark

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python benchmarks/run_benchmark.py
```

Defaults atuais:

```text
dataset: clustered
dimensions: 1536
k: 3
queries por tamanho: 50
clusters: 32
cluster spread: 0.35

n:
1.000
5.000
10.000
25.000
50.000
```

Para testar escalas maiores:

```bash
python benchmarks/run_benchmark.py --sizes 50000 100000 250000
```

Como esta é uma implementação manual de HNSW em Python, escalas grandes podem ter custo elevado de build e memória.

Para executar o controle com vetores independentes:

```bash
python benchmarks/run_benchmark.py --dataset random
```

Para alterar a estrutura dos clusters:

```bash
python benchmarks/run_benchmark.py \
  --clusters 64 \
  --cluster-spread 0.45
```

Quanto maior `cluster-spread`, mais dispersos ficam os vetores ao redor do centro de cada grupo.

## Métricas

O benchmark mede:

- p50 de latência;
- p95 de latência;
- média de latência;
- tempo de construção;
- variação aproximada de RSS;
- **Recall@3** contra a busca exata;
- speedup de p95.

### Recall@3

A busca exata funciona como ground truth.

Se o Top-3 exato for:

```text
[10, 25, 90]
```

E HNSW retornar:

```text
[10, 25, 44]
```

então HNSW encontrou dois dos três vizinhos exatos:

```text
Recall@3 = 2 / 3 = 0.6667
```

Essa métrica impede concluir que HNSW é melhor apenas porque é mais rápido. O objetivo é observar o equilíbrio entre ganho de latência e perda de recall.

## O que este benchmark não tenta reproduzir

Ele não simula integralmente:

- PostgreSQL/PgVector;
- ACLs e filtros SQL reais;
- threshold de relevância da aplicação;
- deduplicação final de chunks;
- contextualização da conversa;
- custo da chamada ao serviço de embedding;
- IVFFlat.

Esses componentes influenciam a latência ponta a ponta, mas a intenção deste repositório é isolar a comparação algorítmica entre **busca cosine exata** e **HNSW**.

O código da aplicação analisada possui criação de IVFFlat durante ingestão, porém o uso desse índice pela consulta principal não foi confirmado apenas por leitura do código. Por isso, IVFFlat não foi incluído como baseline experimental aqui.

## Resultados

Os resultados são escritos em:

```text
benchmarks/results/results.csv
```

Campos adicionais registram o tipo de dataset, número de clusters e dispersão usada.

## Gráficos

Depois do benchmark:

```bash
python benchmarks/plot_results.py
```

Isso gera:

```text
benchmarks/results/latency_p95.png
benchmarks/results/recall_at_k.png
```

O gráfico de recall usa automaticamente o `k` registrado no CSV; com a configuração padrão, ele exibe **Recall@3**.

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

## Interpretação esperada

A conclusão da Spike não deve ser "HNSW é melhor" de forma genérica.

O formato mais útil é:

> Até aproximadamente N candidatos, a busca exata apresentou custo aceitável. A partir de X, HNSW reduziu a latência p95 em Y mantendo Recall@3 de Z.

Isso torna explícito o ponto de crossover e o custo de aproximação do algoritmo.
