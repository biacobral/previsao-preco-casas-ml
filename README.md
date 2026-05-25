# 🏠 Previsão de Preços de Casas — King County, WA

Projeto de Machine Learning para previsão de preços de imóveis no Condado de King, Washington (EUA), desenvolvido como projeto de iniciação em Machine Learning e IA.

---

## 📁 Estrutura do Projeto

```
├── datasets/
│   ├── kc_house_data.csv           # Dados físicos dos imóveis (21.613 registros)
│   ├── zipcode_demographics.csv    # Perfil socioeconômico por CEP (70 CEPs)
│   ├── future_unseen_examples.csv  # Imóveis sem preço — alvo de predição
│   └── future_predictions.csv      # Resultado das predições (gerado pelo modelo)
├── proposta.ipynb                  # Notebook principal com todo o pipeline
└── relatorio_house_prices.docx     # Relatório técnico do projeto
```

---

## 🎯 Objetivo

Construir um modelo preditivo capaz de estimar o preço de venda de imóveis a partir de suas características físicas e do perfil socioeconômico do CEP onde estão localizados.

---

## 🗂️ Datasets

| Dataset | Registros | Descrição |
|---|---|---|
| `kc_house_data.csv` | 21.613 | Dados físicos: área, quartos, banheiros, qualidade construtiva, localização, data de venda |
| `zipcode_demographics.csv` | 70 | Dados por CEP: renda mediana, escolaridade, perfil urbano/rural |
| `future_unseen_examples.csv` | 100 | Imóveis sem preço para predição final |

Os datasets são combinados via **left join** pela chave `zipcode`, preservando todos os imóveis do dataset principal.

---

## ⚙️ Pipeline

### 1. Pré-processamento
- Merge dos dados físicos com os demográficos por `zipcode`
- Engenharia de features:
  - `renovated` — flag booleana derivada de `yr_renovated`
  - `basement` — flag booleana derivada de `sqft_basement`
  - `dif_sqft_living` — diferença entre a área da casa e a média dos 15 vizinhos
  - `dif_sqft_lot` — diferença entre o lote da casa e a média dos 15 vizinhos
- Remoção de colunas com data leakage (`hous_val_amt`), identificadores e redundâncias
- Tratamento do outlier `bedrooms=33` via **KNNImputer** (k=5)

### 2. Análise de Outliers
- Detecção por **Z-score** (|z| > 3) e **IQR** (1.5×IQR)
- Análise restrita a variáveis contínuas com semântica mensurável
- Decisão baseada em domínio: outliers de imóveis de luxo foram mantidos

### 3. Modelo — MLP (Keras/TensorFlow)

```
Input (26 features)
    → Dense(64, relu)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)          # saída: preço em USD
```

| Parâmetro | Valor |
|---|---|
| Optimizer | Adam |
| Loss | MAE |
| Batch size | 32 |
| Epochs | 100 |
| Validation split | 10% |

### 4. Generalização
- **Train/Test Split** 80/20 com `random_state=42`
- **StandardScaler** fitado apenas no treino, aplicado via `transform` no teste e nos dados futuros
- **Validation split** para monitoramento da curva de perda durante o treino

---

## 📊 Resultados

| Métrica | Valor |
|---|---|
| MAE (Erro Absoluto Médio) | ~$100.000 |
| R² (Coeficiente de Determinação) | ~0.78 |

O modelo explica ~78% da variância dos preços. O erro médio de $100k representa aproximadamente 22% do preço mediano ($450.000) — aceitável considerando que o modelo não utiliza dados de comparação de mercado.

---

## 🔑 Features Selecionadas

| Categoria | Features |
|---|---|
| Físicas | `sqft_living`, `grade`, `sqft_above`, `sqft_basement`, `bathrooms`, `bedrooms`, `floors`, `condition`, `waterfront`, `view`, `basement`, `sqft_lot`, `dif_sqft_living`, `dif_sqft_lot` |
| Temporais | `yr_built`, `renovated` |
| Geográficas / Demográficas | `lat`, `long`, `medn_incm_per_prsn_amt`, `medn_hshld_incm_amt`, `per_prfsnl`, `per_bchlr`, `per_urbn`, `per_assoc`, `per_9_to_12`, `per_hsd` |

---

## 🛠️ Tecnologias

- Python 3
- Pandas / NumPy
- Scikit-learn (KNNImputer, StandardScaler, train_test_split)
- TensorFlow / Keras
- Matplotlib / Seaborn

---
