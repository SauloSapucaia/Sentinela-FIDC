#%%
#  MOTOR SENTINELA V0.4  —  PIPELINE COMPLETO DE FRAUDE
#  Autor  : Saulo Sapucaia
#  Versão : 0.4 
#  Fluxo  : Ingestão → EDA → Pré-Proc → Join → APIs → Features → Regras → ML → Score → Exportação → Sumário Executivo
# ==============================================================================

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

plt.style.use('default')

# Paleta de cores centralizada
CORES = {
    'critico': '#c0392b', 'alto': '#e67e22', 'medio': '#f1c40f',
    'baixo': '#27ae60',   'neutro': '#2980b9', 'fundo': '#ecf0f1'
}

print("  🚀  MOTOR SENTINELA V3.0  —  FRAUD DETECTION & MLOps PIPELINE")

#%%
#  ETAPA 1 — INGESTÃO DE DADOS
# ==============================================================================
print("\n[1/10] 📥  Carregando bases de dados...")

df_boletos = pd.read_csv(r'C:\Users\saulo\OneDrive\Documentos\5_Faculdade\0_Conteudos\ENTERPRISE CHALLENGE\base_boletos_fiap.csv', encoding='latin-1') 
df_auxiliar = pd.read_csv(r"C:\Users\saulo\OneDrive\Documentos\5_Faculdade\0_Conteudos\ENTERPRISE CHALLENGE\base_auxiliar_fiap.csv", encoding='latin-1')

# Conversão de datas já na ingestão para uso imediato no EDA
for col in ['dt_emissao', 'dt_vencimento', 'dt_pagamento']:
    df_boletos[col] = pd.to_datetime(df_boletos[col], errors='coerce')

print(f"  Boletos : {len(df_boletos):} registros | {df_boletos.shape[1]} colunas")
print(f"  Auxiliar: {len(df_auxiliar):} registros | {df_auxiliar.shape[1]} colunas")
match_pag = df_boletos['id_pagador'].isin(df_auxiliar['id_cnpj']).mean() * 100
print(f"  Match pagadores na base auxiliar: {match_pag:.2f}%")

#%%
#  ETAPA 2 — Exploração dos Dados (EDA)
# ==============================================================================
print("\n[2/10] 🔍  Análise Exploratória Profunda (Deep EDA)...")

# QUALIDADE DOS DADOS — Mapa de nulos
print("  EDA BLOCO 1 — QUALIDADE DOS DADOS")

def mapa_nulos(df, nome):
    resumo = pd.DataFrame({
        'Campo':    df.columns,
        'Nulos':    df.isnull().sum().values,
        'Pct_Nulo': (df.isnull().sum() / len(df) * 100).values,
    }).sort_values('Pct_Nulo', ascending=False)
    resumo = resumo[resumo['Nulos'] > 0]
    print(f"\n  {nome} — campos com dados ausentes:")
    for _, r in resumo.iterrows():
        barra  = '█' * int(r['Pct_Nulo'] / 5) + '-' * (20 - int(r['Pct_Nulo'] / 5))
        alerta = '⚠️  CRÍTICO' if r['Pct_Nulo'] > 30 else ('⚡ ATENÇÃO' if r['Pct_Nulo'] > 5 else '  ok')
        print(f"    {r['Campo']:<45} {barra}  {r['Pct_Nulo']:5.1f}%  {alerta}")
    return resumo

mapa_nulos(df_boletos,  "BASE BOLETOS")
mapa_nulos(df_auxiliar, "BASE AUXILIAR")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (df, nome) in zip(axes, [(df_boletos, 'Boletos'), (df_auxiliar, 'Auxiliar')]):
    pcts       = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    cores_bar  = [CORES['critico'] if v > 30 else CORES['alto'] if v > 5 else CORES['baixo'] for v in pcts]
    pcts.plot(kind='barh', ax=ax, color=cores_bar, edgecolor='white')
    ax.set_title(f'% de Nulos — Base {nome}', fontweight='bold')
    ax.set_xlabel('% de valores ausentes')
    for i, v in enumerate(pcts):
        if v > 0:
            ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8)
    ax.set_xlim(0, max(pcts) * 1.25 + 1)
plt.tight_layout()
plt.show()


# DUPLICATAS TRANSACIONAIS
print("  EDA BLOCO 2 — DUPLICATAS TRANSACIONAIS")

CHAVE_TRANSAC   = ['id_pagador', 'id_beneficiario', 'vlr_nominal', 'dt_emissao']
mask_dup        = df_boletos.duplicated(subset=CHAVE_TRANSAC, keep=False)
df_duplicatas   = df_boletos[mask_dup].copy()
n_grupos_dup    = (df_boletos.groupby(CHAVE_TRANSAC)
                             .filter(lambda x: len(x) > 1)[CHAVE_TRANSAC]
                             .drop_duplicates().shape[0])

analise_dup = df_duplicatas.groupby(CHAVE_TRANSAC).agg(
    qtd_boletos            = ('id_boleto',   'count'),
    tipos_baixa_distintos  = ('tipo_baixa',  'nunique'),
    tipos_baixa            = ('tipo_baixa',  lambda x: ' | '.join(x.dropna().unique()[:3]))
).reset_index()

dup_baixa_diff  = analise_dup[analise_dup['tipos_baixa_distintos'] > 1]
dup_baixa_igual = analise_dup[analise_dup['tipos_baixa_distintos'] == 1]
vlr_total_dup   = df_duplicatas.groupby(CHAVE_TRANSAC)['vlr_nominal'].first().sum()


print(f"  Boletos em grupos duplicados : {len(df_duplicatas)} ({len(df_duplicatas)/len(df_boletos)*100:.1f}% da base)")
print(f"  Grupos únicos de duplicata   : {n_grupos_dup}")
print(f"  ├─ Com tipo_baixa DIFERENTE (mais suspeito): {len(dup_baixa_diff):,}")
print(f"  └─ Com tipo_baixa IGUAL (cobrança dupla)   : {len(dup_baixa_igual):,}")
print(f"  Valor nominal total exposto                : R$ {vlr_total_dup:.0f}")

print("\n  Grupos de duplicatas com tipos de baixa DIFERENTES (sinal forte de fraude):")
print(dup_baixa_diff.sort_values('qtd_boletos', ascending=False)[['qtd_boletos', 'vlr_nominal', 'tipos_baixa']].head(5).to_string(index=False))

print("\n  Grupos de duplicatas com tipos de baixa IGUAIS (cobrança dupla clássica):")
print(dup_baixa_igual.sort_values('qtd_boletos', ascending=False)[['qtd_boletos', 'vlr_nominal', 'tipos_baixa']].head(5).to_string(index=False))

# OVERPAYMENTS
print("  EDA BLOCO 3 — OVERPAYMENTS (vlr_baixa > vlr_nominal)")

df_pagos_raw = df_boletos[df_boletos['vlr_baixa'].notna() & (df_boletos['vlr_baixa'] > 0)].copy()
df_pagos_raw['diff_pct'] = ((df_pagos_raw['vlr_baixa'] - df_pagos_raw['vlr_nominal']) / df_pagos_raw['vlr_nominal']) * 100

overpay  = df_pagos_raw[df_pagos_raw['diff_pct'] > 1]
underpay = df_pagos_raw[df_pagos_raw['diff_pct'] < -5]

print(f"  Total boletos com valor de baixa registrado: {len(df_pagos_raw)}")
print(f"  ├─ Overpayments (> +1%)                    : {len(overpay)} ({len(overpay)/len(df_pagos_raw)*100:.1f}%)")
print(f"  │    Maior overpayment                     : +{overpay['diff_pct'].max():.1f}%")
print(f"  └─ Underpayments (< -5%)                   : {len(underpay)} ({len(underpay)/len(df_pagos_raw)*100:.1f}%)")
print(f"       Maior desconto                        : {underpay['diff_pct'].min():.1f}%")
print("\n  Overpayments por tipo_baixa:")
print(overpay['tipo_baixa'].value_counts().to_string())


# Gráfico 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
dados_plot = df_pagos_raw[(df_pagos_raw['diff_pct'] > -15) & (df_pagos_raw['diff_pct'] < 20)]['diff_pct']
ax.hist(dados_plot, bins=60, color=CORES['neutro'], edgecolor='white', alpha=0.8)
for v, lbl, cor in [(0, 'Valor exato', 'black'), (1, '+1% alerta', CORES['alto']),
                    (10, '+10% crítico', CORES['critico'])]:
    ax.axvline(v, color=cor, linewidth=1.5, linestyle='--', label=lbl)
ax.set_title('% Diferença entre Baixa e Nominal', fontweight='bold')
ax.set_xlabel('% (positivo = overpayment)')
ax.legend(fontsize=8)

ax = axes[1]
tipos_princ = df_pagos_raw['tipo_baixa'].value_counts().head(4).index
grupos      = [df_pagos_raw[df_pagos_raw['tipo_baixa'] == t]['diff_pct'].values for t in tipos_princ]
ax.boxplot(grupos, labels=[t[:28] for t in tipos_princ], patch_artist=True,
           boxprops=dict(facecolor=CORES['fundo']),
           medianprops=dict(color=CORES['critico'], linewidth=2))
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_title('Diferença % por Tipo de Baixa', fontweight='bold')
ax.tick_params(axis='x', rotation=28)
plt.tight_layout()
plt.show()

# ANÁLISE TEMPORAL E SAZONALIDADE
print("  EDA BLOCO 4 — ANÁLISE TEMPORAL E SAZONALIDADE")

df_boletos['mes_emissao']  = df_boletos['dt_emissao'].dt.to_period('M')
df_boletos['dias_ate_venc'] = (df_boletos['dt_vencimento'] - df_boletos['dt_emissao']).dt.days
df_boletos['dias_atraso']   = (df_boletos['dt_pagamento']  - df_boletos['dt_vencimento']).dt.days

serie_mensal     = df_boletos['mes_emissao'].value_counts().sort_index()
pct_top2_meses   = serie_mensal.nlargest(2).sum() / len(df_boletos) * 100
boletos_pre2023  = df_boletos[(df_boletos['dt_emissao'] < '2023-01-01') & (df_boletos['dt_pagamento'] > '2024-01-01')].shape[0]
mask_vlr_baixa_nulo = df_boletos['vlr_baixa'].isnull()

print(f"\n  Período da base                                : {df_boletos['dt_emissao'].min().date()} → {df_boletos['dt_emissao'].max().date()}")
print(f"  Mês de maior emissão                           : {serie_mensal.idxmax()} ({serie_mensal.max():,} boletos)")
print(f"  2 meses concentram                             : {pct_top2_meses:.1f}% da base")
print(f"  Boletos emitidos pré-2023 com pagamento em 2024: {boletos_pre2023}")
print(f"  Vencimento no mesmo dia da emissão (prazo=0)   : {(df_boletos['dias_ate_venc'] == 0).sum()}")
print(f"  Vencimento com prazo > 1 ano                   : {(df_boletos['dias_ate_venc'] > 365).sum()}")
print(f"  Prazo máximo encontrado                        : {df_boletos['dias_ate_venc'].max():.0f} dias")
print(f"  Atrasos negativos (pagamento antes do vencimento): {(df_boletos['dias_atraso'] < 0).sum()}")
print(f"  Atrasos > 1 ano                                : {(df_boletos['dias_atraso'] > 365).sum()}")
print(f"  Atraso máximo encontrado                       : {df_boletos['dias_atraso'].max():.0f} dias")
print(f"  Boletos com pagamento registrado mas sem data de baixa: {mask_vlr_baixa_nulo.sum()} (possível sinal de pagamento fantasma ou erro de registro)")

# Gráfico 3
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
cores_b = [CORES['critico'] if v == serie_mensal.max() else CORES['neutro'] for v in serie_mensal]
serie_mensal.plot(kind='bar', ax=axes[0, 0], color=cores_b, edgecolor='white')
axes[0, 0].set_title('Volume de Emissão Mensal', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

prazo_ok = df_boletos['dias_ate_venc'].dropna()
prazo_ok[prazo_ok < 400].hist(bins=50, ax=axes[0, 1], color=CORES['neutro'], edgecolor='white')
axes[0, 1].axvline(30,  color=CORES['alto'],    linestyle='--', label='30 dias')
axes[0, 1].axvline(180, color=CORES['critico'], linestyle='--', label='180 dias')
axes[0, 1].set_title('Prazo Emissão → Vencimento (dias)', fontweight='bold')
axes[0, 1].legend()

atr = df_boletos['dias_atraso'].dropna()
atr[(atr > -30) & (atr < 365)].hist(bins=60, ax=axes[1, 0], color=CORES['medio'], edgecolor='white', alpha=0.8)
axes[1, 0].axvline(0, color='black', linewidth=2, linestyle='--', label='Vencimento exato')
axes[1, 0].set_title('Atraso no Pagamento (dias)', fontweight='bold')
axes[1, 0].legend()

ultimos_meses = serie_mensal[serie_mensal > 50].tail(4).index
df_temp       = df_boletos[df_boletos['mes_emissao'].isin(ultimos_meses)]
grupos_mes    = [df_temp[df_temp['mes_emissao'] == m]['vlr_nominal'].values for m in ultimos_meses]
axes[1, 1].boxplot(grupos_mes, labels=[str(m) for m in ultimos_meses], patch_artist=True,
                   boxprops=dict(facecolor=CORES['fundo']),
                   medianprops=dict(color=CORES['critico'], linewidth=2), showfliers=False)
axes[1, 1].set_title('Valor Nominal por Mês (sem outliers)', fontweight='bold')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
plt.suptitle('ANÁLISE TEMPORAL — BASE BOLETOS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# PERFIL FINANCEIRO DOS CNPJs (BASE AUXILIAR)
print("  EDA BLOCO 5 — PERFIL FINANCEIRO DOS CNPJs")

p75_atraso  = df_auxiliar['media_atraso_dias'].quantile(0.75)
alto_inad   = (df_auxiliar['share_vl_inad_pag_bol_6_a_15d'] > 0.5).sum()
faixas_liq  = pd.cut(df_auxiliar['sacado_indice_liquidez_1m'].dropna(),
                      bins=[0, 0.3, 0.5, 0.7, 1.01],
                      labels=['Crítica (<0.3)', 'Baixa (0.3–0.5)', 'Média (0.5–0.7)', 'Boa (>0.7)'])

pct_ced_nulo = df_auxiliar['cedente_indice_liquidez_1m'].isnull().mean() * 100

print(f"\n  Perfil Financeiro dos CNPJs na Base Auxiliar:")
print(f"  Índice de Liquidez do Sacado (1 mês) — Mediana: {df_auxiliar['sacado_indice_liquidez_1m'].median():.2f} | P50: {df_auxiliar['sacado_indice_liquidez_1m'].quantile(0.5):.2f} | P75: {df_auxiliar['sacado_indice_liquidez_1m'].quantile(0.75):.2f}")
print(f"media_atraso_dias — Mediana: {df_auxiliar['media_atraso_dias'].median():.0f}d | P75: {p75_atraso:.0f}d | Máx: {df_auxiliar['media_atraso_dias'].max():.0f}d")
print(f"CNPJs com >50% do valor em atraso 6-15d: {alto_inad}")
print(f"Segmentação por liquidez (sacado 1m):")
print(faixas_liq.value_counts().sort_index().to_string())

# Gráfico 4
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
df_auxiliar['sacado_indice_liquidez_1m'].dropna().hist(
    bins=40, ax=axes[0, 0], color=CORES['neutro'], edgecolor='white')
for v, cor, lbl in [(0.3, CORES['critico'], '<0.3'), (0.5, CORES['alto'], '<0.5')]:
    axes[0, 0].axvline(v, color=cor, linestyle='--', linewidth=2, label=f'Crítica {lbl}')
axes[0, 0].set_title('Liquidez do Sacado (1 mês)', fontweight='bold')
axes[0, 0].legend()

df_auxiliar['cedente_indice_liquidez_1m'].dropna().hist(
    bins=40, ax=axes[0, 1], color=CORES['medio'], edgecolor='white')
axes[0, 1].axvline(0.3, color=CORES['critico'], linestyle='--', linewidth=2)
axes[0, 1].set_title(
    f'Liquidez do Cedente (1 mês)\n⚠️ {pct_ced_nulo:.0f}% são imputados', fontweight='bold')

df_auxiliar['media_atraso_dias'].dropna().hist(
    bins=50, ax=axes[1, 0], color=CORES['alto'], edgecolor='white', alpha=0.8)
axes[1, 0].axvline(df_auxiliar['media_atraso_dias'].median(), color='black', linestyle='--',
                    linewidth=2, label=f'Mediana: {df_auxiliar["media_atraso_dias"].median():.0f}d')
axes[1, 0].axvline(p75_atraso, color=CORES['critico'], linestyle='--',
                    linewidth=2, label=f'P75: {p75_atraso:.0f}d')
axes[1, 0].set_title('Média de Atraso Histórico (dias)', fontweight='bold')
axes[1, 0].legend()

uf_counts = df_auxiliar['uf'].fillna('NI').value_counts().head(12)
cores_uf  = [CORES['critico'] if u == 'NI' else CORES['neutro'] for u in uf_counts.index]
uf_counts.plot(kind='barh', ax=axes[1, 1], color=cores_uf, edgecolor='white')
axes[1, 1].set_title('CNPJs por Estado (UF)', fontweight='bold')
plt.suptitle('PERFIL FINANCEIRO — BASE AUXILIAR', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# DISTRIBUIÇÃO DE VALOR, ASSIMETRIA E CORRELAÇÕES
print("  EDA BLOCO 6 — DISTRIBUIÇÃO DE VALOR, ASSIMETRIA E CORRELAÇÕES")

assimetria = df_boletos['vlr_nominal'].skew()
q99 = df_boletos['vlr_nominal'].quantile(0.99)
q01 = df_boletos['vlr_nominal'].quantile(0.01)

print(f"Assimetria (skew) vlr_nominal    : {assimetria:.2f}  → distribuição fortemente direcionada à direita")
print(f"Boletos acima do P99 (R$ {q99:.0f}) : {(df_boletos['vlr_nominal'] > q99).sum()}")
print(f"Boletos abaixo do P01 (R$ {q01:.2f}) : {(df_boletos['vlr_nominal'] < q01).sum()}")
print("\nCorrelações numéricas com vlr_nominal (base boletos):")
print(df_boletos.corr(numeric_only=True)['vlr_nominal'].sort_values(ascending=False).to_string())

# Gráfico 5
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
limite_95 = df_boletos['vlr_nominal'].quantile(0.95)
df_boletos[df_boletos['vlr_nominal'] < limite_95]['vlr_nominal'].hist(
    bins=50, ax=axes[0], color=CORES['neutro'], edgecolor='white', alpha=0.8)
axes[0].set_title(f'Distribuição Original (corte P95 = R${limite_95:,.0f})', fontweight='bold')
axes[0].set_xlabel('Valor Nominal (R$)')

np.log1p(df_boletos['vlr_nominal']).hist(
    bins=50, ax=axes[1], color='purple', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribuição Normalizada (log1p)', fontweight='bold')
axes[1].set_xlabel('log(1 + Valor Nominal)')
plt.tight_layout()
plt.show()

#%%
# CONCENTRAÇÃO E REDE DE RELACIONAMENTOS
print("  EDA BLOCO 7 — CONCENTRAÇÃO E REDE DE RELACIONAMENTOS")

top_pag = df_boletos['id_pagador'].value_counts()
top_ben = df_boletos['id_beneficiario'].value_counts()
p80_pag = (top_pag.cumsum() / top_pag.sum() <= 0.8).sum()
p80_ben = (top_ben.cumsum() / top_ben.sum() <= 0.8).sum()

par_contagem    = df_boletos.groupby(['id_pagador', 'id_beneficiario']).agg(
    qtd=('id_boleto', 'count'), vlr_total=('vlr_nominal', 'sum')).reset_index()
pares_repetidos = par_contagem[par_contagem['qtd'] > 10]

print(f" Pagadores únicos     : {df_boletos['id_pagador'].nunique():,}")
print(f" → 80% dos boletos vêm de apenas {p80_pag} pagadores")
print(f" → Top pagador tem {top_pag.iloc[0]:,} boletos")

print(f"\n Beneficiários únicos : {df_boletos['id_beneficiario'].nunique():,}")
print(f" → 80% dos boletos vêm de apenas {p80_ben} beneficiários")

print(f"\n Pares pagador-beneficiário com > 10 transações: {len(pares_repetidos):,}")
print(f" Par mais frequente  : {pares_repetidos['qtd'].max():,} boletos | R$ {pares_repetidos.sort_values('qtd', ascending=False)['vlr_total'].iloc[0]:,.0f}")

# Gráfico 6
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lorenz = top_pag.sort_values().cumsum() / top_pag.sum()
x      = np.linspace(0, 1, len(lorenz))
axes[0].plot(x, lorenz.values, color=CORES['critico'], linewidth=2, label='Lorenz')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Ideal')
axes[0].fill_between(x, lorenz.values, x, alpha=0.2, color=CORES['critico'])
axes[0].set_title('Concentração por Pagador (Lorenz)', fontweight='bold')
axes[0].legend()

top15_pares = par_contagem.nlargest(15, 'qtd')
cores_pares = [CORES['critico'] if v > 50 else CORES['alto'] if v > 25 else CORES['neutro']
               for v in top15_pares['qtd']]
axes[1].barh(range(15), top15_pares['qtd'].values, color=cores_pares, edgecolor='white')
axes[1].set_yticks(range(15))
axes[1].set_yticklabels([f'Par {i+1}' for i in range(15)], fontsize=8)
axes[1].set_title('Top 15 Pares Pagador–Beneficiário', fontweight='bold')
axes[1].set_xlabel('Qtd Boletos')
plt.tight_layout()
plt.show()

#%%
#  ETAPA 3 — PRÉ-PROCESSAMENTO E IMPUTAÇÃO
# ==============================================================================
print("\n[3/10]  Pré-Processamento e Imputação...")

# 3a) Salvar máscaras ANTES de qualquer fillna (usadas nas novas flags)
mask_vlr_baixa_nulo = df_boletos['vlr_baixa'].isna()
mask_cedente_sem_historico = df_auxiliar['cedente_indice_liquidez_1m'].isna()

# 3b) Flag de duplicata transacional (calculada com dados brutos)
df_boletos['flag_duplicata_transacional'] = ( df_boletos.duplicated(subset=CHAVE_TRANSAC, keep=False).astype(int) )

# 3c) Limpeza boletos
df_boletos['vlr_baixa'] = df_boletos['vlr_baixa'].fillna(0)

# 3d) Limpeza auxiliar — categoria
df_auxiliar['uf'] = df_auxiliar['uf'].fillna('NI')

# 3e) Flag cedente sem histórico (antes de imputar!)
df_auxiliar['flag_cedente_sem_historico'] = mask_cedente_sem_historico.astype(int)

# 3f) Imputação estatística — mediana para variáveis contínuas
mediana_liquidez_ced = df_auxiliar['cedente_indice_liquidez_1m'].median()
df_auxiliar['cedente_indice_liquidez_1m'] = ( df_auxiliar['cedente_indice_liquidez_1m'].fillna(mediana_liquidez_ced) )
for col in df_auxiliar.select_dtypes(include=[np.number]).columns:
    df_auxiliar[col] = df_auxiliar[col].fillna(df_auxiliar[col].median())

print(f"  ✅ Flag duplicata_transacional criada ({df_boletos['flag_duplicata_transacional'].sum():,} boletos)")
print(f"Flag duplicata criada | ")
print(f"Cedentes sem histórico imputados: {mask_cedente_sem_historico.sum()}")
print(f"  ✅ Imputação por mediana aplicada em {df_auxiliar.shape[1]} colunas numéricas")


#%%
#  ETAPA 4 — ENRIQUECIMENTO (DUPLO JOIN)
# ==============================================================================
print("\n[4/10] 🔗  Enriquecimento via Duplo Join...")
# O join duplo traz informações tanto do pagador quanto do beneficiário, permitindo análises comparativas e relações entre os
df_sentinela = pd.merge( df_boletos, df_auxiliar, left_on='id_pagador', right_on='id_cnpj', how='left' )
df_sentinela = pd.merge( df_sentinela, df_auxiliar, left_on='id_beneficiario', right_on='id_cnpj', how='left', suffixes=('_pagador', '_beneficiario'))

print(f"  ✅ DataFrame enriquecido: {df_sentinela.shape[0]:,} linhas | {df_sentinela.shape[1]} colunas")


#%%
#  ETAPA 5 — CONSUMO DE APIs EXTERNAS
# ==============================================================================
print("\n[5/10] 🌐  Consumindo APIs Externas...")

# API IBGE — Setores CNAE (fator de risco setorial)
try:
    dados_ibge = requests.get( "https://servicodados.ibge.gov.br/api/v2/cnae/divisoes" ).json()
    df_setores = pd.DataFrame( [ {'divisao_cnae': int(d['id']), 'setor_oficial_ibge': d['descricao']} for d in dados_ibge] )
    df_sentinela['divisao_cnae'] = ( df_sentinela['cd_cnae_prin_pagador'].fillna(0).astype(str).str.zfill(7).str[:2].astype(int) )
    df_sentinela = pd.merge(df_sentinela, df_setores, on='divisao_cnae', how='left' )
    df_sentinela['setor_oficial_ibge'] = ( df_sentinela['setor_oficial_ibge'].fillna('Setor Desconhecido') )
    print("  ✅ API IBGE (CNAE): OK")
except Exception as e:
    df_sentinela['setor_oficial_ibge'] = 'Setor Desconhecido'
    print(f"  ⚠️  API IBGE: FALHOU ({e})")

# API BACEN — Taxa Selic (fator de risco macro) 
fator_risco_macro = 0
try:
    selic_atual = float(requests.get( "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json").json()[0]['valor'])
    fator_risco_macro = 5 if selic_atual > 10.0 else 0
    print(f"  ✅ API BACEN (Selic): {selic_atual}% → Fator de risco macro: +{fator_risco_macro} pts")
except Exception as e:
    print(f"  ⚠️  API BACEN: FALHOU ({e}) — fator_risco_macro = 0")


#%%
#  ETAPA 6 — FEATURE ENGINEERING & Z-SCORE SETORIAL
# ==============================================================================
print("\n[6/10] ⚙️   Feature Engineering: Temporais, Comportamentais e Z-Score Setorial...")

# 0) Ordenação temporal
df_sentinela = df_sentinela.sort_values(by=['id_pagador', 'dt_emissao'])

# 1) HISTÓRICO DO PAGADOR
# ------------------------------------------------------------------------------
# quantidade de boletos emitidos por pagador
df_sentinela['qtd_boletos_pagador'] = ( df_sentinela.groupby('id_pagador')['id_boleto'].transform('count') )
# ticket médio do pagador
df_sentinela['ticket_medio_pagador'] = ( df_sentinela.groupby('id_pagador')['vlr_nominal'].transform('mean') )
# desvio do boleto atual vs histórico do pagador
df_sentinela['desvio_ticket_pagador'] = ( df_sentinela['vlr_nominal'] / df_sentinela['ticket_medio_pagador'] )


# 2) HISTÓRICO DO BENEFICIÁRIO
# ------------------------------------------------------------------------------
# quantidade de boletos recebidos por beneficiário
df_sentinela['qtd_boletos_beneficiario'] = ( df_sentinela.groupby('id_beneficiario')['id_boleto'].transform('count') )
# ticket médio do beneficiário
df_sentinela['ticket_medio_beneficiario'] = ( df_sentinela.groupby('id_beneficiario')['vlr_nominal'].transform('mean') )
# desvio do boleto atual vs histórico do beneficiário
df_sentinela['desvio_ticket_beneficiario'] = ( df_sentinela['vlr_nominal'] / df_sentinela['ticket_medio_beneficiario'] )


# 3) RELAÇÃO ENTRE A DUPLA PAGADOR x BENEFICIÁRIO
# ------------------------------------------------------------------------------
df_sentinela['freq_dupla_pag_benef'] = ( df_sentinela.groupby(['id_pagador', 'id_beneficiario'])['id_boleto'].transform('count') )
df_sentinela['ticket_medio_dupla'] = ( df_sentinela.groupby(['id_pagador', 'id_beneficiario'])['vlr_nominal'].transform('mean') )


# 4) CONCENTRAÇÃO POR BENEFICIÁRIO
# ------------------------------------------------------------------------------
total_por_benef = ( df_sentinela.groupby('id_beneficiario')['vlr_nominal'].transform('sum') )
total_geral = df_sentinela['vlr_nominal'].sum()
df_sentinela['share_financeiro_benef'] = total_por_benef / total_geral


# 5) PADRÃO TEMPORAL
# ------------------------------------------------------------------------------
# intervalo entre boletos do mesmo pagador
df_sentinela['dias_desde_ultimo_boleto_pagador'] = (df_sentinela.groupby('id_pagador')['dt_emissao'].diff().dt.days.fillna(999) )
# boletos emitidos no mesmo dia pela mesma empresa
df_sentinela['qtd_mesmo_dia_pagador'] = (df_sentinela.groupby(['id_pagador', 'dt_emissao'])['id_boleto'].transform('count') )


# 6) VALOR REPETIDO
# ------------------------------------------------------------------------------
df_sentinela['freq_mesmo_valor_pagador'] = (df_sentinela.groupby(['id_pagador', 'vlr_nominal'])['id_boleto'].transform('count'))
df_sentinela['freq_mesmo_valor_benef'] = (df_sentinela.groupby(['id_beneficiario', 'vlr_nominal'])['id_boleto'].transform('count') )


# 7) ANOMALIA POR GRUPO
# ------------------------------------------------------------------------------
# z-score por UF
uf_stats = (df_sentinela.groupby('uf_pagador')['vlr_nominal'].agg(['mean', 'std']).reset_index())
uf_stats.columns = ['uf_pagador', 'media_uf', 'std_uf']
# para evitar divisão por zero, preencher std_uf com 1 onde for zero ou nulo
df_sentinela = df_sentinela.merge(uf_stats, on='uf_pagador', how='left')
df_sentinela['std_uf'] = df_sentinela['std_uf'].fillna(1)
df_sentinela['z_score_uf'] = ((df_sentinela['vlr_nominal'] - df_sentinela['media_uf']) /df_sentinela['std_uf'] )

# 8) VARIÁVEIS TEMPORAIS
# ------------------------------------------------------------------------------
df_sentinela['dias_emissao_ate_pagamento'] = ( (df_sentinela['dt_pagamento'] - df_sentinela['dt_emissao']).dt.days.fillna(-1) )
df_sentinela['dias_atraso_real'] = ( (df_sentinela['dt_pagamento'] - df_sentinela['dt_vencimento']).dt.days.fillna(0) )
df_sentinela['antiguidade_boleto'] = ( (df_sentinela['dt_emissao'].dt.to_period('M').apply(lambda p: p.ordinal) - df_boletos['mes_emissao'].min().ordinal) )


# 9) TRANSFORMAÇÃO LOGARÍTMICA
# ------------------------------------------------------------------------------
df_sentinela['vlr_nominal_log'] = np.log1p(df_sentinela['vlr_nominal'])


# 10) CLUSTERS DE VALOR
# ------------------------------------------------------------------------------
# Clusters de valor (quartis)
labels_clusters = ['1 - Baixo', '2 - Médio', '3 - Alto', '4 - Muito Alto']
df_sentinela['cluster_valor_nominal'], bins_valor = pd.qcut( df_sentinela['vlr_nominal'], q=4, labels=labels_clusters, retbins=True)

# 11) DIFERENÇA ENTRE NOMINAL E BAIXA (OVERPAYMENT)
# ------------------------------------------------------------------------------
df_sentinela['diff_vlr_pago'] = df_sentinela['vlr_nominal'] - df_sentinela['vlr_baixa']
df_sentinela['diff_vlr_pago_pct'] = ( (df_sentinela['vlr_baixa'] - df_sentinela['vlr_nominal']) / df_sentinela['vlr_nominal'].replace(0, np.nan) * 100).fillna(0)

# 12) Z-Score setorial (valor vs média do CNAE do beneficiário)
# ------------------------------------------------------------------------------
cnae_stats = (df_sentinela.groupby('cd_cnae_prin_beneficiario')['vlr_nominal']
              .agg(['mean', 'std']).reset_index()
              .rename(columns={'mean': 'vlr_medio_cnae', 'std': 'vlr_std_cnae'}))
df_sentinela = pd.merge(df_sentinela, cnae_stats, on='cd_cnae_prin_beneficiario', how='left')
df_sentinela['vlr_std_cnae']    = df_sentinela['vlr_std_cnae'].fillna(1)
df_sentinela['z_score_setorial'] = np.where( df_sentinela['vlr_std_cnae'] > 0, 
                                            (df_sentinela['vlr_nominal'] - df_sentinela['vlr_medio_cnae']) / df_sentinela['vlr_std_cnae'], 0)

# 13) FLAGS COMPORTAMENTAIS (baseadas nas features criadas)
df_sentinela['flag_valor_fora_padrao_pagador'] = np.where( df_sentinela['desvio_ticket_pagador'] > 5, 1, 0 )
df_sentinela['flag_concentracao_beneficiario'] = np.where( df_sentinela['share_financeiro_benef'] > 0.10, 1, 0 )
df_sentinela['flag_repeticao_mesmo_valor'] = np.where( df_sentinela['freq_mesmo_valor_benef'] >= 5, 1, 0 )
df_sentinela['flag_emissao_em_lote'] = np.where( df_sentinela['qtd_mesmo_dia_pagador'] >= 4, 1, 0 )
df_sentinela['flag_uf_anomala'] = np.where( df_sentinela['z_score_uf'] > 3, 1, 0 )

print("  ✅ Feature engineering aplicada: 20+ novas variáveis criadas, incluindo z-scores setorial e por UF, padrões temporais e comportamentais, e clusters de valor.")


#%%
#  ETAPA 7 — MOTOR DE REGRAS SENTINELA V3
# ==============================================================================
print("\n[7/10] 🚨  Motor de Regras Sentinela V3...")

# TABELA DE PESOS — cada critério e seu peso na pontuação de risco
PESOS_RISCO = {
    # Regras clássicas
    'boleto_zumbi':              50,   # boleto > 1 ano com liquidez baixa
    'pagamento_fantasma':        45,   # baixa interbancária sem vlr_baixa original
    'duplicata_transacional':    40,   # mesmo título com id_boleto diferente
    'anomalia_setorial':         35,   # z-score valor > 3 no CNAE do beneficiário
    'colusao':                   30,   # mesmo CNAE + UF, pagador ≠ beneficiário
    'titulo_frio':               25,   # pago em até 1 dia, valor alto
    'toxicidade_dupla':          25,   # liquidez baixa em ambos os lados
    'overpayment_alto':          20,   # pago >10% acima do nominal
    'historico_atraso_critico':  15,   # media_atraso_dias > P75
    # Regras comportamentais 
    'concentracao_beneficiario': 25,   # 1 beneficiário capta >10% do volume total
    'emissao_em_lote':           20,   # ≥4 boletos emitidos no mesmo dia pelo pagador
    'valor_fora_padrao_pagador': 20,   # ticket 5x acima da média histórica do pagador
    'uf_anomala':                20,   # z-score por UF > 3 (análogo ao setorial)
    'repeticao_mesmo_valor':     15,   # mesmo valor ≥5x para o mesmo beneficiário

}

# TABELA DE LIMITES — faixas de classificação final
LIMITES_CLASSIFICACAO = {
    'critico': 60,
    'alto':    35,
    'medio':   10,
}

# Inicializar pontuação com fator macro
df_sentinela['pontuacao_risco'] = fator_risco_macro

# Flag: BOLETO ZUMBI 
df_sentinela['flag_boleto_zumbi'] = np.where(
    (df_sentinela['dias_emissao_ate_pagamento'] > 365) &
    (df_sentinela['sacado_indice_liquidez_1m_pagador'] < 0.30) &
    (df_sentinela['vlr_nominal'] > bins_valor[2]),
    1, 0)

# Flag: PAGAMENTO FANTASMA (regra corrigida pós-EDA)
df_sentinela['flag_pagamento_fantasma'] = np.where(
    (df_sentinela['tipo_baixa'].str.contains('interbancaria', case=False, na=False)) &
    (df_sentinela['id_boleto'].isin(df_boletos[mask_vlr_baixa_nulo]['id_boleto'])),
    1, 0)

# Flag: DUPLICATA TRANSACIONAL 
df_sentinela['flag_duplicata_transacional'] = df_sentinela['flag_duplicata_transacional'].fillna(0)

# Flag: ANOMALIA SETORIAL 
df_sentinela['flag_anomalia_setorial'] = np.where(
    df_sentinela['z_score_setorial'] > 3, 1, 0)

# Flag: COLUSÃO 
df_sentinela['flag_colusao'] = np.where(
    (df_sentinela['cd_cnae_prin_pagador'] == df_sentinela['cd_cnae_prin_beneficiario']) &
    (df_sentinela['uf_pagador'] == df_sentinela['uf_beneficiario']) &
    (df_sentinela['id_pagador'] != df_sentinela['id_beneficiario']),
    1, 0)

# Flag: TÍTULO FRIO 
df_sentinela['flag_titulo_frio'] = np.where(
    (df_sentinela['dias_emissao_ate_pagamento'] >= 0) &
    (df_sentinela['dias_emissao_ate_pagamento'] <= 1) &
    (df_sentinela['vlr_nominal'] > bins_valor[2]),
    1, 0)

# Flag: TOXICIDADE DUPLA
df_sentinela['flag_toxicidade_dupla'] = np.where(
    (df_sentinela['sacado_indice_liquidez_1m_pagador'] < 0.40) &
    (df_sentinela['indicador_liquidez_quantitativo_3m_beneficiario'] < 0.40),
    1, 0)

# Flag: OVERPAYMENT ALTO
df_sentinela['flag_overpayment_alto'] = np.where( df_sentinela['diff_vlr_pago_pct'] > 10, 1, 0 )

# Flag: HISTÓRICO DE ATRASO CRÍTICO
df_sentinela['flag_historico_atraso_critico'] = np.where( df_sentinela['media_atraso_dias_pagador'] > p75_atraso, 1, 0)

# Acumulando pontuação 
for flag, peso in PESOS_RISCO.items():
    df_sentinela['pontuacao_risco'] += df_sentinela[f'flag_{flag}'] * peso

# Resumo das flags
print("  Flags ativas — taxas de disparo:")
todas_flags = [f'flag_{k}' for k in PESOS_RISCO.keys()]
for flag in todas_flags:
    taxa = df_sentinela[flag].mean() * 100
    alerta = " ⚠️ ALTA" if taxa > 5 else (" 🔴 ZERO" if taxa == 0 else "")
    print(f"    {flag:<40} {taxa:5.2f}%{alerta}")


#%%
#  ETAPA 8 — MACHINE LEARNING (ISOLATION FOREST APRIMORADO)
# ==============================================================================
print("\n[8/10] 🤖  Treinando Isolation Forest (features aprimoradas)...")

features_iforest = [
    # Valor e liquidez (base) 
    'vlr_nominal',
    'dias_emissao_ate_pagamento',
    'dias_atraso_real',
    'sacado_indice_liquidez_1m_pagador',
    'indicador_liquidez_quantitativo_3m_beneficiario',
    'media_atraso_dias_pagador',
    'diff_vlr_pago_pct',
    # Histórico do pagador 
    'qtd_boletos_pagador',
    'ticket_medio_pagador',
    'desvio_ticket_pagador',
    # Histórico do beneficiário
    'qtd_boletos_beneficiario',
    'ticket_medio_beneficiario',
    'desvio_ticket_beneficiario',
    # Relação pagador x beneficiário
    'freq_dupla_pag_benef',
    'ticket_medio_dupla',
    # Concentração financeira
    'share_financeiro_benef',
    # Padrão temporal
    'dias_desde_ultimo_boleto_pagador',
    'qtd_mesmo_dia_pagador',
    # Repetição de valor
    'freq_mesmo_valor_pagador',
    'freq_mesmo_valor_benef',
    # Anomalia estatística multidimensional
    'z_score_setorial',
    'z_score_uf',
]

# Preparação da matriz
X_iforest = df_sentinela[features_iforest].copy()
X_iforest = X_iforest.replace([np.inf, -np.inf], np.nan).fillna(0)

# Winsorização (P1–P99): reduz distorção de valores extremos absurdos
for col in X_iforest.columns:
    X_iforest[col] = X_iforest[col].clip( lower=X_iforest[col].quantile(0.01), upper=X_iforest[col].quantile(0.99) )

# Cálculo de uma métrica de "sinais de fraude" para comparação com a IA, baseada na média das flags clássicas e anomalias estatísticas
sinais_fraude = (
    df_sentinela['flag_duplicata_transacional'].mean() +
    df_sentinela['flag_overpayment_alto'].mean() +
    df_sentinela['flag_boleto_zumbi'].mean() +
    df_sentinela['flag_concentracao_beneficiario'].mean() +
    (df_sentinela['z_score_setorial'] > 3).mean()
) / 5

contamination_rate = max(0.008, min(0.025, sinais_fraude * 1.8))
print(f"  → Contamination rate adaptativo calculado: {contamination_rate:.3f} ({sinais_fraude*100:.1f}% sinais fortes na EDA)")

# RobustScaler: mais resistente a outliers do que o StandardScaler, preservando a estrutura dos dados para o Isolation Forest.
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_iforest)

pca = PCA(n_components=min(15, X_iforest.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  → PCA aplicado: {X_pca.shape[1]} componentes explicam {pca.explained_variance_ratio_.sum():.1%} da variância")

# Isolation Forest (melhorado)
modelo_if = IsolationForest(
    n_estimators=300,
    contamination=contamination_rate,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
modelo_if.fit(X_pca)

# Local Outlier Factor 
modelo_lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination_rate,
    n_jobs=-1
)
lof_scores = modelo_lof.fit_predict(X_pca)

df_sentinela['flag_fraude_if'] = np.where(modelo_if.predict(X_pca) == -1, 1, 0)
df_sentinela['flag_fraude_lof'] = np.where(lof_scores == -1, 1, 0)
df_sentinela['flag_fraude_ia'] = np.where( (df_sentinela['flag_fraude_if'] == 1) | (df_sentinela['flag_fraude_lof'] == 1), 1, 0 )
df_sentinela['score_anomalia_ia'] = modelo_if.decision_function(X_pca)  
df_sentinela['pontuacao_risco'] += df_sentinela['flag_fraude_ia'] * 35

print(f"  ✅ Ensemble IF + LOF treinado")
print(f"     Anomalias IF   : {df_sentinela['flag_fraude_if'].sum():,} ({df_sentinela['flag_fraude_if'].mean()*100:.2f}%)")
print(f"     Anomalias LOF  : {df_sentinela['flag_fraude_lof'].sum():,} ({df_sentinela['flag_fraude_lof'].mean()*100:.2f}%)")
print(f"     Anomalias FINAl (ensemble): {df_sentinela['flag_fraude_ia'].sum():,} ({df_sentinela['flag_fraude_ia'].mean()*100:.2f}%)")

# Perfil das anomalias detectadas pela IA
anomalias = df_sentinela[df_sentinela['flag_fraude_ia'] == 1]
print("\n  Perfil estatístico das anomalias IA:")
print(anomalias[[ 'vlr_nominal', 'desvio_ticket_pagador', 'freq_dupla_pag_benef', 'share_financeiro_benef', 'qtd_mesmo_dia_pagador']].describe().round(3).to_string())

# Explicabilidade simples 
_features_explic = [
    'vlr_nominal', 'desvio_ticket_pagador', 'desvio_ticket_beneficiario',
    'freq_dupla_pag_benef', 'share_financeiro_benef',
    'qtd_mesmo_dia_pagador', 'dias_desde_ultimo_boleto_pagador',
    'z_score_setorial', 'z_score_uf'
]
_refs = {col: {'p95': df_sentinela[col].quantile(0.95),
               'p99': df_sentinela[col].quantile(0.99),
               'p05': df_sentinela[col].quantile(0.05)}
               for col in _features_explic}

# A função _explicar() gera uma explicação textual simples para cada anomalia detectada, destacando quais features estão mais fora do padrão em relação aos percentis de referência.
def _explicar(linha):
    motivos = []
    for col in _features_explic:
        v, r = linha[col], _refs[col]
        if v >= r['p99']:
            motivos.append(f"{col} extremamente alto")
        elif v >= r['p95']:
            motivos.append(f"{col} acima do padrão")
        elif col == 'dias_desde_ultimo_boleto_pagador' and v <= r['p05']:
            motivos.append(f"{col} extremamente baixo")
    return " | ".join(motivos) if motivos else "combinação multivariada fora do padrão"

df_sentinela['motivo_anomalia_ia'] = np.where( df_sentinela['flag_fraude_ia'] == 1, df_sentinela.apply(_explicar, axis=1), 'não aplicável')

print("\n  Top 8 motivos de anomalia IA:")
print(df_sentinela[df_sentinela['flag_fraude_ia'] == 1]['motivo_anomalia_ia'].value_counts().head(8).to_string())


#%%
#  ETAPA 9 — SCORE FINAL E EXPORTAÇÃO DO CSV
# ==============================================================================
print("\n[9/10] 💾  Classificação Final e Exportação...")

condicoes_risco = [
    df_sentinela['pontuacao_risco'] >= LIMITES_CLASSIFICACAO['critico'],
    (df_sentinela['pontuacao_risco'] >= LIMITES_CLASSIFICACAO['alto']) &
    (df_sentinela['pontuacao_risco'] <  LIMITES_CLASSIFICACAO['critico']),
    (df_sentinela['pontuacao_risco'] >= LIMITES_CLASSIFICACAO['medio']) &
    (df_sentinela['pontuacao_risco'] <  LIMITES_CLASSIFICACAO['alto']),
    df_sentinela['pontuacao_risco'] < LIMITES_CLASSIFICACAO['medio'],
]
df_sentinela['classificacao_sentinela'] = np.select(
    condicoes_risco,
    ['CRÍTICO (Fraude Forte)', 'ALTO RISCO', 'MÉDIO RISCO', 'BAIXO RISCO'],
    default='INDEFINIDO')

# Colunas do dashboard Power BI
COLUNAS_DASHBOARD = [
    'id_boleto', 'dt_emissao', 'dt_vencimento', 'dt_pagamento',
    'vlr_nominal', 'vlr_baixa', 'tipo_especie',
    'cd_cnae_prin_pagador', 'uf_pagador', 'setor_oficial_ibge',
    'cluster_valor_nominal', 'z_score_setorial', 'diff_vlr_pago_pct',
    'pontuacao_risco', 'classificacao_sentinela',
    # Flags de regras
    'flag_pagamento_fantasma',    'flag_anomalia_setorial',  'flag_colusao',
    'flag_titulo_frio',           'flag_toxicidade_dupla',   'flag_boleto_zumbi',
    'flag_duplicata_transacional','flag_overpayment_alto',   'flag_historico_atraso_critico',
    # Flags comportamentais 
    'flag_concentracao_beneficiario', 'flag_emissao_em_lote',
    'flag_valor_fora_padrao_pagador', 'flag_uf_anomala',
    'flag_repeticao_mesmo_valor',
    # Machine Learning
    'flag_fraude_if',             # flag do Isolation Forest 
    'flag_fraude_lof',            # flag do Local Outlier Factor
    'flag_fraude_ia',             # flag binária (0/1)
    'score_anomalia_ia',          # score contínuo: quanto mais negativo, mais anômalo
    'motivo_anomalia_ia'          # explicação textual do principal vetor de anomalia
]
df_sentinela[COLUNAS_DASHBOARD].to_csv(r"C:\Users\saulo\OneDrive\Documentos\5_Faculdade\3_Trabalhos\Enterprise Challenge\Sentinela_MVP_Final.csv", index=False, encoding='latin-1')

print("\n  Distribuição da classificação final:")
resumo_class = (df_sentinela.groupby('classificacao_sentinela')['vlr_nominal']
                .agg(qtd='count', vlr_medio='mean', vlr_total='sum')
                .sort_values('vlr_total', ascending=False))
resumo_class['vlr_total'] = resumo_class['vlr_total'].map('R$ {:,.0f}'.format)
resumo_class['vlr_medio'] = resumo_class['vlr_medio'].map('R$ {:,.0f}'.format)
print(resumo_class.to_string())
print("\n  ✅ Exportado: Sentinela_MVP_Final.csv")


#%%
#  ETAPA 10 — SUMÁRIO EXECUTIVO PARA A BANCA
# ==============================================================================
print("\n[10/10] 📊  Gerando Sumário Executivo para a Banca...")

df_criticos = df_sentinela[df_sentinela['classificacao_sentinela'] == 'CRÍTICO (Fraude Forte)']
df_zumbis   = (df_sentinela[df_sentinela['flag_boleto_zumbi'] == 1]
               .sort_values('vlr_nominal', ascending=False))

# ── Gráfico 7: Diagnóstico de flags (movido do EDA Bloco 6) ──────────────────
todas_flags_final = [f'flag_{k}' for k in PESOS_RISCO.keys()] + ['flag_fraude_ia']
taxa_flags        = df_sentinela[todas_flags_final].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
taxa_plot  = taxa_flags.sort_values()
cores_flag = [CORES['critico'] if v == 0 else CORES['critico'] if v > 10 else CORES['neutro']
              for v in taxa_plot]
taxa_plot.plot(kind='barh', ax=axes[0], color=cores_flag, edgecolor='white')
axes[0].set_yticklabels(
    [l.replace('flag_', '').replace('_', ' ').title() for l in taxa_plot.index])
axes[0].set_title('Taxa de Ativação por Flag (%)', fontweight='bold')
for i, v in enumerate(taxa_plot):
    axes[0].text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=8)
axes[0].set_xlim(0, taxa_plot.max() * 1.3)

pontuacao = df_sentinela['pontuacao_risco']
axes[1].hist(pontuacao[pontuacao < 100], bins=50, color=CORES['neutro'], edgecolor='white', alpha=0.8)
for lim, nome, cor in [(60, 'CRÍTICO', CORES['critico']),
                        (35, 'ALTO',    CORES['alto']),
                        (10, 'MÉDIO',   CORES['medio'])]:
    axes[1].axvline(lim, color=cor, linestyle='--', linewidth=2, label=f'{nome} (≥{lim})')
axes[1].set_title('Distribuição da Pontuação de Risco Final', fontweight='bold')
axes[1].set_xlabel('Pontuação de Risco')
axes[1].legend()
plt.tight_layout()
plt.show()

# ── Gráfico 8: Anatomia dos casos críticos ───────────────────────────────────
FLAGS_NOMES = {
    'flag_pagamento_fantasma':        'Pagamento Fantasma (rev.)',
    'flag_duplicata_transacional':    'Duplicata Transacional',
    'flag_anomalia_setorial':         'Anomalia Setorial (Z-Score CNAE)',
    'flag_colusao':                   'Colusão (Auto-Faturamento)',
    'flag_titulo_frio':               'Título Frio',
    'flag_toxicidade_dupla':          'Toxicidade Dupla',
    'flag_boleto_zumbi':              'Boleto Zumbi (>1 ano)',
    'flag_overpayment_alto':          'Overpayment Alto (>10%)',
    'flag_historico_atraso_critico':  'Histórico Atraso Crítico',
    'flag_concentracao_beneficiario': 'Concentração Beneficiário (>10%)',
    'flag_emissao_em_lote':           'Emissão em Lote (≥4/dia)',
    'flag_valor_fora_padrao_pagador': 'Valor Fora Padrão Pagador (>5x)',
    'flag_uf_anomala':                'Anomalia por UF (Z-Score)',
    'flag_repeticao_mesmo_valor':     'Repetição Mesmo Valor (≥5x)',
    'flag_fraude_ia':                 'Anomalia IA (Isolation Forest)',
}
contagens = [df_criticos[col].sum() for col in FLAGS_NOMES.keys()]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bars = axes[0].barh(list(FLAGS_NOMES.values()), contagens,
                     color=CORES['critico'], edgecolor='black', alpha=0.85)
axes[0].set_title('Anatomia dos Boletos Críticos', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Quantidade de Ocorrências')
axes[0].invert_yaxis()
for bar in bars:
    axes[0].text(bar.get_width() + (max(contagens) * 0.02),
                  bar.get_y() + bar.get_height() / 2,
                  f'{int(bar.get_width())}', va='center', fontsize=10, fontweight='bold')
axes[0].set_xlim(0, max(contagens) * 1.2 if max(contagens) > 0 else 1)

analise_uf = (df_criticos['uf_pagador'].value_counts().head(10).reset_index())
analise_uf.columns = ['UF', 'Qtd']
axes[1].barh(analise_uf['UF'], analise_uf['Qtd'],
              color=CORES['alto'], edgecolor='black', alpha=0.85)
axes[1].set_title('Casos Críticos por Estado (UF Pagador)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Quantidade de Casos Críticos')
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

# ── Resumo numérico para a banca ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("  📋  SUMÁRIO EXECUTIVO — MOTOR SENTINELA V3.0")
print("=" * 80)

vlr_critico = df_criticos['vlr_nominal'].sum()
vlr_total   = df_sentinela['vlr_nominal'].sum()


print(f" BASE ANALISADA:")
print(f" ├─ Total de boletos              : {len(df_sentinela):,}")
print(f" ├─ Valor nominal total           : R$ {vlr_total:,.0f}")
print(f" ├─ Período                       : {df_boletos['dt_emissao'].min().date()} → {df_boletos['dt_emissao'].max().date()}")
print(f" ├─ Duplicatas transacionais      : {df_sentinela['flag_duplicata_transacional'].sum():>6} boletos ({df_sentinela['flag_duplicata_transacional'].mean()*100:.1f}%)")
print(f" ├─ Overpayments altos (>10%)     : {df_sentinela['flag_overpayment_alto'].sum():>6} boletos")
print(f" └─ Histórico atraso crítico      : {df_sentinela['flag_historico_atraso_critico'].sum():>6} boletos")

print(f"\n CLASSIFICAÇÃO FINAL:")
print(f" ├─ CRÍTICO (Fraude Forte)        : {(df_sentinela['classificacao_sentinela']=='CRÍTICO (Fraude Forte)').sum():>6} boletos  | R$ {vlr_critico:>15,.0f}")
print(f" ├─ ALTO RISCO                    : {(df_sentinela['classificacao_sentinela']=='ALTO RISCO').sum():>6} boletos")
print(f" ├─ MÉDIO RISCO                   : {(df_sentinela['classificacao_sentinela']=='MÉDIO RISCO').sum():>6} boletos")
print(f" └─ BAIXO RISCO                   : {(df_sentinela['classificacao_sentinela']=='BAIXO RISCO').sum():>6} boletos")

print(f" MACHINE LEARNING ENSEMBLE (IF + LOF):")
print(f" └─ Anomalias detectadas         : {df_sentinela['flag_fraude_ia'].sum():>6} boletos")

print(f"\n BOLETOS ZUMBI:")
print(f" ├─ Total                         : {len(df_zumbis):>6}")
print(f" └─ Top 3 por valor:")
if len(df_zumbis) > 0:
    print(df_zumbis[['id_boleto', 'vlr_nominal', 'dias_emissao_ate_pagamento',
                       'sacado_indice_liquidez_1m_pagador']].head(3).to_string(index=False))


print(f"\n ANATOMIA DOS CASOS CRÍTICOS:")
for flag, nome in FLAGS_NOMES.items():
    qtd = df_criticos[flag].sum()
    taxa = (qtd / len(df_criticos) * 100) if len(df_criticos) > 0 else 0
    print(f" ├─ {nome:<35} : {qtd:>6} boletos ({taxa:.1f}%)")

print(f"\n Perfil por estado (UF) dos casos críticos - Top 5 UFs:")
analise_uf_full = df_criticos['uf_pagador'].value_counts().reset_index()
analise_uf_full.columns = ['Estado (UF)', 'Qtd Críticos']
analise_uf_full['%'] = (analise_uf_full['Qtd Críticos'] / len(df_criticos) * 100).map('{:.1f}%'.format)
print(analise_uf_full.head(5).to_string(index=False))

print("=" * 80)
print("  ✅  PIPELINE CONCLUÍDO — Artefatos prontos para o Power BI")
print("=" * 80)

#%%