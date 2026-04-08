# 🚀 Sentinela V3.0 — Auditoria Preditiva & MLOps para FIDC

**Detecção inteligente de fraudes em boletos com Deep EDA, Motor de Regras + Ensemble IA (Isolation Forest + LOF) e dashboard executivo 100% interativo.**

Projeto final do **FIAP Enterprise Challenge 2025/2026** — Núclea.

---

## 📋 Sobre o Projeto

A **Sentinela V3.0** é um pipeline completo de detecção de fraudes em direitos creditórios (FIDC).  
Transforma mais de **7.118 boletos** em uma carteira limpa e rentável, bloqueando **924 boletos críticos** (R$ 87,8 Mi preservados) e destacando **49,1% da carteira como segura para investimento**.

Combina:
- **Deep EDA** com insights acionáveis
- **Motor de Regras** com +15 flags comportamentais
- **Ensemble de Machine Learning** (Isolation Forest + Local Outlier Factor)
- **Explainable AI (XAI)** com justificativas textuais automáticas
- **Dashboard executivo** 100% client-side (HTML + Tailwind + Chart.js)

---

## ✨ Principais Funcionalidades

### Dashboard Executivo (index.html)
- Storytelling completo do projeto (Problema → Solução → Resultado)
- KPIs dinâmicos em tempo real
- Gráficos interativos (Anatomia das Fraudes, Geografia de Risco, Classificação Final, Anomalia Temporal, etc.)
- Tabela de ameaças com ordenação, filtros e explicabilidade IA
- **Recomendações de Investimento** (top boletos de baixo risco)
- Modais ricos: Deep EDA + Motor IA completo com código Python

### Motor Python (Sentinela_FIDC.py)
- Pipeline end-to-end (Ingestão → EDA → Features → Regras → ML → Export)
- Feature Engineering avançado (Z-Score setorial/UF, padrões temporais, concentração, etc.)
- Contamination rate **adaptativo** baseado na EDA
- Explicabilidade textual automática
- Exporta `Sentinela_MVP_Final.csv` pronto para o dashboard

---

## 🛠️ Tecnologias Utilizadas

| Camada          | Tecnologia                          |
|-----------------|-------------------------------------|
| **Backend**     | Python 3 + Pandas + scikit-learn   |
| **Machine Learning** | Isolation Forest + LOF + PCA + RobustScaler |
| **Frontend**    | HTML5 + Tailwind CSS + Chart.js + PapaParse |
| **Visualização**| Chart.js + Font Awesome            |
| **Dados**       | CSV (client-side)                  |
| **APIs**        | IBGE (CNAE) + BACEN (Selic)        |

**Zero dependências de servidor** — o dashboard roda direto no navegador.

---

## 🚀 Como Executar

### Opção 1 — Dashboard (mais simples)
1. Baixe o repositório
2. Certifique-se de que o arquivo **`Sentinela_MVP_Final.csv`** esteja na mesma pasta que `index.html`
3. Abra o arquivo **`index.html`** no navegador (Chrome recomendado)

Pronto! Tudo carrega automaticamente.

### Opção 2 — Reexecutar o Motor Python
```bash
pip install pandas numpy scikit-learn matplotlib requests
python Sentinela_FIDC.py
