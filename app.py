# ============================================================
# ANÁLISE DE INTELIGÊNCIA COMPUTACIONAL — REGRESSÃO E CLASSIFICAÇÃO
# Interface gráfica moderna com Streamlit + Plotly (interativo)
# Autor: Bruno Barbosa
# ============================================================

import io, os, json, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   OrdinalEncoder, LabelEncoder)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.utils.discovery import all_estimators
from scipy.stats import loguniform, uniform

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

RANDOM_STATE = 42

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análise IC — Regressão e Classificação",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2a5298;
        padding: 15px; border-radius: 8px; margin: 5px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white; border: none; border-radius: 6px;
        padding: 10px 24px; font-weight: 600;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #2a5298, #1e3c72); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ESTADO DA SESSÃO
# ============================================================
def _init_state():
    defaults = {
        'df_raw': None, 'df_encoded': None, 'df_modeling': None,
        'results': None, 'task_type': 'regression',
        'encoding_ready': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

# ============================================================
# CLASSES E FUNÇÕES AUXILIARES
# ============================================================
class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.10, max_features=None):
        self.threshold = threshold; self.max_features = max_features
    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        y_sr = pd.Series(y)
        if not pd.api.types.is_numeric_dtype(y_sr):
            y_sr = pd.Series(LabelEncoder().fit_transform(y_sr.astype(str)))
        else:
            y_sr = y_sr.astype(float)
        corr = X_df.apply(lambda c: c.corr(y_sr)).abs().fillna(0)
        ordered = corr.sort_values(ascending=False)
        if self.max_features is not None:
            chosen = list(ordered.head(int(self.max_features)).index)
        else:
            chosen = list(ordered[ordered >= self.threshold].index)
        if not chosen: chosen = [ordered.index[0]]
        self.selected_indices_ = [X_df.columns.get_loc(c) for c in chosen]
        self.selected_features_ = [str(self.feature_names_in_[i]) for i in self.selected_indices_]
        return self
    def transform(self, X): return np.asarray(X)[:, self.selected_indices_]
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def detect_categorical_columns(df, max_unique_int=20):
    cats = []
    for col in df.columns:
        try:
            if (pd.api.types.is_object_dtype(df[col]) or
                pd.api.types.is_string_dtype(df[col]) or
                pd.api.types.is_categorical_dtype(df[col]) or
                pd.api.types.is_bool_dtype(df[col]) or
                str(df[col].dtype) in ('category', 'string')):
                cats.append(col); continue
            if pd.api.types.is_integer_dtype(df[col]):
                nun = df[col].nunique(dropna=True)
                if 1 < nun <= max_unique_int: cats.append(col)
        except Exception: continue
    return cats


def _coerce_numeric(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): continue
        if not (pd.api.types.is_object_dtype(df[c]) or
                pd.api.types.is_string_dtype(df[c]) or
                str(df[c].dtype) in ('string',)): continue
        s = df[c].astype(str).str.strip()
        s_num = s.str.replace(r'R\$\s*', '', regex=True).str.replace(r'%\s*$', '', regex=True).str.replace(' ', '')
        conv_d = pd.to_numeric(s_num, errors='coerce')
        s_br = s_num.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        conv_b = pd.to_numeric(s_br, errors='coerce')
        nn = df[c].notna().sum()
        if nn == 0: continue
        if conv_d.notna().sum() / nn >= 0.9: df[c] = conv_d
        elif conv_b.notna().sum() / nn >= 0.9: df[c] = conv_b
    return df


def load_dataframe(uploaded_file, source_type, uci_id=None):
    if source_type == 'UCI':
        if fetch_ucirepo is None:
            raise ImportError("ucimlrepo não instalado.")
        ds = fetch_ucirepo(id=int(uci_id))
        df = (pd.concat([ds.data.features, ds.data.targets], axis=1)
              if ds.data.features is not None and ds.data.targets is not None
              else ds.data.original)
        return _coerce_numeric(df)
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        raw = uploaded_file.read().decode('utf-8', errors='replace').lstrip('\ufeff')
        first = raw.splitlines()[0] if raw else ''
        cnt = {';': first.count(';'), ',': first.count(','),
               '\t': first.count('\t'), '|': first.count('|')}
        sep = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else ','
        df = pd.read_csv(io.StringIO(raw), sep=sep)
    elif name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    elif name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError('Formato não suportado.')
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    return _coerce_numeric(df)


def apply_categorical_encoding(df, encoding_map, ordinal_orders=None, manual_maps=None):
    ordinal_orders = ordinal_orders or {}; manual_maps = manual_maps or {}
    df_out = df.copy(); relatorio = []
    for col, metodo in encoding_map.items():
        if col not in df_out.columns: continue
        if metodo == 'drop':
            df_out = df_out.drop(columns=[col]); relatorio.append((col, 'drop', '—')); continue
        s = df_out[col].astype(str).replace({'nan': 'missing', 'None': 'missing'})
        df_out[col] = s
        if metodo == 'onehot':
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            arr = enc.fit_transform(df_out[[col]])
            novas = [f'{col}__{c}' for c in enc.categories_[0]]
            df_oh = pd.DataFrame(arr, columns=novas, index=df_out.index)
            df_out = pd.concat([df_out.drop(columns=[col]), df_oh], axis=1)
            relatorio.append((col, 'onehot', f'{len(novas)} colunas'))
        elif metodo == 'label':
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_out[col] = enc.fit_transform(df_out[[col]]).ravel()
            relatorio.append((col, 'label', '1 coluna'))
        elif metodo == 'ordinal':
            ordem = ordinal_orders.get(col, sorted(s.unique().tolist()))
            enc = OrdinalEncoder(categories=[ordem], handle_unknown='use_encoded_value', unknown_value=-1)
            df_out[col] = enc.fit_transform(df_out[[col]]).ravel()
            relatorio.append((col, 'ordinal', f'ordem'))
        elif metodo == 'manual':
            mapa = manual_maps.get(col, {})
            df_out[col] = s.map(mapa).fillna(-1).astype(int)
            relatorio.append((col, 'manual', f'{len(mapa)} mapas'))
    return df_out, relatorio


def iqr_outlier_summary(df):
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, up = q1 - 1.5*iqr, q3 + 1.5*iqr
        n = int(((s < lo) | (s > up)).sum())
        rows.append([col, q1, q3, iqr, lo, up, n, 100*n/len(s)])
    return pd.DataFrame(rows, columns=['Variável','Q1','Q3','IQR','Limite inf.','Limite sup.','Outliers','%'])


def training_outlier_filter(X, y, mode='remove_train'):
    X_df = X.copy(); y_sr = y.copy()
    if mode == 'keep': return X_df, y_sr, 0
    bounds = {}
    for col in X_df.columns:
        s = X_df[col].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5*iqr, q3 + 1.5*iqr)
    if mode == 'remove_train':
        mask = pd.Series(True, index=X_df.index)
        for col in X_df.columns:
            lo, up = bounds.get(col, (-np.inf, np.inf))
            mask &= X_df[col].between(lo, up, inclusive='both') | X_df[col].isna()
        removed = int((~mask).sum())
        return X_df.loc[mask].copy(), y_sr.loc[mask].copy(), removed
    if mode == 'winsorize':
        for col in X_df.columns:
            lo, up = bounds.get(col, (-np.inf, np.inf))
            X_df[col] = X_df[col].clip(lower=lo, upper=up)
        return X_df, y_sr, 0
    if mode == 'median':
        for col in X_df.columns:
            lo, up = bounds.get(col, (-np.inf, np.inf))
            med = X_df[col].median()
            mask = (X_df[col] < lo) | (X_df[col] > up)
            X_df.loc[mask, col] = med
        return X_df, y_sr, 0
    return X_df, y_sr, 0


# ---------- Regressão ----------
def make_pipeline(estimator, threshold=0.10, max_features=None):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', CorrelationSelector(threshold=threshold, max_features=max_features)),
        ('scaler', StandardScaler()),
        ('regressor', estimator)
    ])


def make_param_distributions():
    return {
        'SVR Linear': {'regressor__C': loguniform(1e-2, 1e2),
                       'regressor__epsilon': uniform(0.001, 0.5)},
        'SVR RBF': {'regressor__C': loguniform(1e-2, 1e2),
                    'regressor__gamma': loguniform(1e-3, 1e1),
                    'regressor__epsilon': uniform(0.001, 0.5)}
    }


def build_regression_models(grids, use_linear=True, use_svr_lin=True, use_svr_rbf=True):
    models = {}
    if use_svr_lin:
        models['SVR Linear'] = {'name': 'SVR Linear', 'estimator': SVR(kernel='linear'),
                                'params': {'regressor__C': grids['C_lin'],
                                           'regressor__epsilon': grids['eps_lin']}}
    if use_svr_rbf:
        models['SVR RBF'] = {'name': 'SVR RBF', 'estimator': SVR(kernel='rbf'),
                             'params': {'regressor__C': grids['C_rbf'],
                                        'regressor__gamma': grids['gamma_rbf'],
                                        'regressor__epsilon': grids['eps_rbf']}}
    if use_linear:
        models['Regressão Linear'] = {'name': 'Regressão Linear',
                                      'estimator': LinearRegression(), 'params': {}}
    return models


def evaluate_regression(X_train, y_train, X_test, y_test, models, threshold=0.10,
                        max_features=None, cv=5, search_method='grid', n_iter=20,
                        random_state=42):
    results = {}
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    progress = st.progress(0); status = st.empty()
    total = len(models)
    for i, (name, cfg) in enumerate(models.items()):
        status.info(f"Treinando **{name}** ({i+1}/{total})...")
        pipe = make_pipeline(clone(cfg['estimator']), threshold, max_features)
        try:
            if cfg['params']:
                if search_method == 'random':
                    dist = make_param_distributions().get(name)
                    search = RandomizedSearchCV(pipe, dist, n_iter=int(n_iter), cv=cv_strategy,
                                                scoring='r2', n_jobs=-1, return_train_score=True,
                                                random_state=random_state, refit=True)
                else:
                    search = GridSearchCV(pipe, cfg['params'], cv=cv_strategy, scoring='r2',
                                          n_jobs=-1, return_train_score=True, refit=True)
                search.fit(X_train, y_train)
                best = search.best_estimator_
                best_params = search.best_params_
                best_cv = float(search.best_score_)
                best_cv_std = float(search.cv_results_['std_test_score'][search.best_index_])
                idx = search.best_index_
                cv_scores = np.array([search.cv_results_[f'split{k}_test_score'][idx] for k in range(cv)])
                grid_table = pd.DataFrame(search.cv_results_).sort_values('rank_test_score').head(10)
            else:
                scores = cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring='r2', n_jobs=-1)
                best = pipe.fit(X_train, y_train)
                best_params = {}; best_cv = float(scores.mean()); best_cv_std = float(scores.std())
                cv_scores = scores; grid_table = None
            pred = best.predict(X_test)
            mse = float(mean_squared_error(y_test, pred))
            results[name] = {
                'best_model': best, 'best_params': best_params,
                'cv_mean': best_cv, 'cv_std': best_cv_std, 'cv_scores': cv_scores,
                'test_r2': float(r2_score(y_test, pred)),
                'test_rmse': float(np.sqrt(mse)),
                'test_mae': float(mean_absolute_error(y_test, pred)),
                'test_mse': mse, 'y_pred': pred, 'grid_table': grid_table
            }
        except Exception as e:
            results[name] = {'error': str(e)}
        progress.progress((i+1)/total)
    status.success("✅ Regressão concluída!")
    progress.empty()
    return results


# ---------- Classificação ----------
@st.cache_data(show_spinner=False)
def get_all_classifiers():
    todos = all_estimators(type_filter="classifier")
    validos = []
    for nome, classe in todos:
        try:
            classe()
            validos.append(nome)
        except Exception:
            continue
    return validos


def evaluate_classifiers(X_train, y_train, X_test, y_test, classifier_names, cv=5,
                        random_state=42, top_n=30):
    results = {}
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    progress = st.progress(0); status = st.empty()
    names_to_run = classifier_names[:top_n]
    total = len(names_to_run)
    todos_dict = dict(all_estimators(type_filter="classifier"))
    for i, name in enumerate(names_to_run):
        status.info(f"Avaliando **{name}** ({i+1}/{total})...")
        try:
            clf_class = todos_dict[name]
            pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler()),
                             ('classifier', clf_class())])
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            cv_res = cross_validate(pipe, X_train, y_train, cv=cv_strategy,
                                    scoring=scoring, n_jobs=-1, error_score='raise')
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            results[name] = {
                'best_model': pipe,
                'cv_acc_mean': float(cv_res['test_accuracy'].mean()),
                'cv_acc_std': float(cv_res['test_accuracy'].std()),
                'cv_acc_scores': cv_res['test_accuracy'].tolist(),
                'cv_f1_scores': cv_res['test_f1_weighted'].tolist(),
                'test_accuracy': acc, 'test_precision': prec,
                'test_recall': rec, 'test_f1': f1, 'y_pred': y_pred
            }
        except Exception as e:
            results[name] = {'error': str(e)[:80]}
        progress.progress((i+1)/total)
    status.success("✅ Classificação concluída!")
    progress.empty()
    return results


# ============================================================
# FUNÇÕES DE BOXPLOT INTERATIVO (PLOTLY)
# ============================================================
def plotly_boxplot_regression(valid):
    """
    Boxplot interativo do R² por dobra da CV para cada modelo de regressão.
    """
    box_rows = []
    for name, res in valid.items():
        for fold, score in enumerate(res['cv_scores'], 1):
            box_rows.append({'Modelo': name, 'Fold': fold, 'R²': float(score)})
    box_df = pd.DataFrame(box_rows)

    fig = px.box(
        box_df, x='Modelo', y='R²', color='Modelo',
        points='all', hover_data=['Fold'],
        title='📦 R² por dobra da Validação Cruzada (interativo)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        height=500, showlegend=False,
        xaxis_title='', yaxis_title='R²',
        yaxis=dict(gridcolor='#EEE'),
        plot_bgcolor='white'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig, box_df


def plotly_boxplot_classification(valid):
    """
    Dois boxplots interativos para classificação:
      1) Métricas por modelo (acc, prec, rec, f1) — dispersão interna
      2) Acurácia por dobra da CV — estabilidade do modelo
    """
    # --- (1) Métricas por modelo ---
    metr_rows = []
    for name, res in valid.items():
        metr_rows.append({'Modelo': name, 'Métrica': 'Acurácia',
                          'Valor': res['test_accuracy']})
        metr_rows.append({'Modelo': name, 'Métrica': 'Precisão',
                          'Valor': res['test_precision']})
        metr_rows.append({'Modelo': name, 'Métrica': 'Recall',
                          'Valor': res['test_recall']})
        metr_rows.append({'Modelo': name, 'Métrica': 'F1',
                          'Valor': res['test_f1']})
    metr_df = pd.DataFrame(metr_rows)

    fig1 = px.box(
        metr_df, x='Métrica', y='Valor', color='Métrica',
        points='all', hover_data=['Modelo'],
        title='📦 Distribuição das métricas entre os classificadores (interativo)',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig1.update_layout(
        height=500, showlegend=False,
        yaxis=dict(range=[0, 1.05], gridcolor='#EEE'),
        plot_bgcolor='white'
    )
    fig1.update_traces(marker=dict(size=8, opacity=0.7))

    # --- (2) Acurácia por dobra da CV (top 15 classificadores) ---
    fold_rows = []
    for name, res in valid.items():
        for i, s in enumerate(res.get('cv_acc_scores', []), 1):
            fold_rows.append({'Classificador': name, 'Fold': i, 'Acurácia': s})
    fold_df = pd.DataFrame(fold_rows)

    fig2 = None
    if not fold_df.empty:
        # Ordena pelos top 15 por média de acurácia
        medias = fold_df.groupby('Classificador')['Acurácia'].mean() \
                        .sort_values(ascending=False)
        top15 = medias.head(15).index.tolist()
        fold_top = fold_df[fold_df['Classificador'].isin(top15)]

        fig2 = px.box(
            fold_top, x='Classificador', y='Acurácia', color='Classificador',
            points='all', hover_data=['Fold'],
            title='📦 Acurácia por dobra da CV — Top 15 (interativo)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_layout(
            height=550, showlegend=False,
            xaxis_title='', yaxis_title='Acurácia',
            xaxis=dict(tickangle=-40),
            yaxis=dict(range=[0, 1.05], gridcolor='#EEE'),
            plot_bgcolor='white'
        )
        fig2.update_traces(marker=dict(size=8, opacity=0.7))

    return fig1, fig2, metr_df, fold_df


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">🧠 Análise de Inteligência Computacional</h1>
    <p style="margin:5px 0 0 0; opacity:0.9;">Regressão e Classificação com Validação Cruzada</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Configuração")
    task_type = st.radio("Tipo de Tarefa",
                         [("Regressão", "regression"), ("Classificação", "classification")],
                         format_func=lambda x: x[0])
    st.session_state['task_type'] = task_type[1]

    st.divider()
    st.subheader("📁 Fonte de Dados")
    source = st.radio("Origem", ["Arquivo Local", "UCI Repository"])

    uploaded_file = None; uci_id = None
    if source == "Arquivo Local":
        uploaded_file = st.file_uploader("Envie o arquivo",
                                          type=['csv', 'xlsx', 'xls', 'json'])
    else:
        uci_id = st.number_input("ID do Dataset (UCI)", min_value=1, value=186, step=1)

    if st.button("📥 Carregar Dados", width='stretch'):
        with st.spinner("Carregando..."):
            try:
                if source == "Arquivo Local" and uploaded_file:
                    df = load_dataframe(uploaded_file, 'file')
                elif source == "UCI Repository" and uci_id:
                    df = load_dataframe(None, 'UCI', uci_id)
                else:
                    st.error("Selecione uma fonte válida.")
                    st.stop()
                st.session_state['df_raw'] = df
                st.session_state['df_encoded'] = None
                st.session_state['df_modeling'] = None
                st.session_state['encoding_ready'] = False
                st.success(f"✅ {df.shape[0]} linhas × {df.shape[1]} colunas")
            except Exception as e:
                st.error(f"Erro: {e}")

# ---------- ÁREA PRINCIPAL ----------
if st.session_state['df_raw'] is None:
    st.info("👈 Use a barra lateral para carregar um conjunto de dados.")
    st.stop()

df_raw = st.session_state['df_raw']

tabs = st.tabs(["📊 Diagnóstico", "🏷️ Codificação", "🎯 Modelagem", "📈 Resultados"])

# ============ TAB 1: DIAGNÓSTICO ============
with tabs[0]:
    st.header("📊 Descrição dos Dados")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas", df_raw.shape[0])
    col2.metric("Colunas", df_raw.shape[1])
    col3.metric("Numéricas", df_raw.select_dtypes(include=np.number).shape[1])
    col4.metric("Categóricas", len(detect_categorical_columns(df_raw)))

    st.subheader("Tipos de dados por coluna")
    info_cols = []
    for c in df_raw.columns:
        info_cols.append({
            'Coluna': c, 'dtype': str(df_raw[c].dtype),
            'Não-nulos': int(df_raw[c].notna().sum()),
            'Nulos': int(df_raw[c].isna().sum()),
            'Únicos': int(df_raw[c].nunique(dropna=True)),
            'Amostra': ' | '.join(df_raw[c].dropna().astype(str).head(3).tolist())[:60]
        })
    st.dataframe(pd.DataFrame(info_cols), width='stretch', height=300)

    with st.expander("📋 Estatísticas descritivas", expanded=True):
        st.dataframe(df_raw.describe().T.round(4), width='stretch')

    with st.expander("🔍 Outliers (IQR)"):
        st.dataframe(iqr_outlier_summary(df_raw).round(4), width='stretch')

    st.subheader("📈 Visualizações")
    num_cols = list(df_raw.select_dtypes(include=np.number).columns)
    if len(num_cols) > 0:
        col1, col2 = st.columns(2)
        with col1:
            n_hist = min(len(num_cols), 4)
            fig, axes = plt.subplots(n_hist, 1, figsize=(6, 2.5*n_hist))
            if n_hist == 1: axes = [axes]
            for ax, c in zip(axes, num_cols[:n_hist]):
                sns.histplot(df_raw[c].dropna(), kde=True, ax=ax, color='steelblue')
                ax.set_title(f'{c}', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with col2:
            if len(num_cols) >= 2:
                corr = df_raw[num_cols].corr()
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(corr, annot=(len(num_cols) <= 10), fmt='.2f',
                            cmap='coolwarm', center=0, square=True, ax=ax,
                            cbar_kws={'shrink': 0.7})
                ax.set_title('Matriz de correlação', fontsize=10)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============ TAB 2: CODIFICAÇÃO ============
with tabs[1]:
    st.header("🏷️ Codificação de Variáveis Categóricas")
    cats = detect_categorical_columns(df_raw)

    if not cats:
        st.success("Nenhuma coluna categórica detectada. Prossiga para a Modelagem.")
        st.session_state['df_modeling'] = df_raw.select_dtypes(include=np.number).copy()
        st.session_state['df_encoded'] = df_raw
        st.session_state['encoding_ready'] = True
    else:
        st.write(f"**Detectadas {len(cats)} colunas categóricas.** Configure abaixo:")
        encoding_map = {}; ordinal_orders = {}; manual_maps = {}

        for col in cats:
            with st.expander(f"🔸 {col}", expanded=False):
                uniq_vals = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                n_uniq = len(uniq_vals)
                c1, c2, c3 = st.columns(3)
                c1.metric("Únicos", n_uniq)
                c2.metric("Ausentes", int(df_raw[col].isna().sum()))
                c3.metric("dtype", str(df_raw[col].dtype)[:15])

                prev = df_raw[col].dropna().astype(str).head(50).tolist()
                st.caption("**Primeiros 50 valores:**")
                st.code(' | '.join(prev[:50]), language=None)

                metodo = st.radio(
                    "Método de codificação:",
                    ['onehot', 'label', 'ordinal', 'manual', 'drop'],
                    format_func=lambda x: {
                        'onehot': 'One-Hot (uma coluna por categoria)',
                        'label': 'Label (0..k-1, ordem alfabética)',
                        'ordinal': 'Ordinal (ordem customizada)',
                        'manual': 'Manual (mapeamento próprio)',
                        'drop': 'Descartar coluna'
                    }[x],
                    key=f'radio_{col}', horizontal=False
                )
                encoding_map[col] = metodo

                if metodo == 'ordinal':
                    ordem = st.text_input("Ordem (separada por vírgula):",
                                          value=', '.join(uniq_vals), key=f'ord_{col}')
                    ordinal_orders[col] = [x.strip() for x in ordem.split(',') if x.strip()]
                elif metodo == 'manual':
                    sug = ', '.join([f'{v}={i}' for i, v in enumerate(uniq_vals)])
                    mapa_txt = st.text_input("Mapeamento (categoria=código):",
                                             value=sug, key=f'map_{col}')
                    mapa = {}
                    for par in mapa_txt.split(','):
                        if '=' in par:
                            k, v = par.split('=', 1)
                            try: mapa[k.strip()] = int(v.strip())
                            except ValueError: pass
                    manual_maps[col] = mapa

        if st.button("✅ Aplicar Codificação", width='stretch', type='primary'):
            with st.spinner("Codificando..."):
                df_enc, relatorio = apply_categorical_encoding(df_raw, encoding_map,
                                                                ordinal_orders, manual_maps)
                st.session_state['df_encoded'] = df_enc
                st.session_state['df_modeling'] = df_enc.select_dtypes(include=np.number).copy()
                st.session_state['encoding_ready'] = True
                st.success(f"✅ Dataset final: {df_enc.shape[0]} linhas × {df_enc.shape[1]} colunas")
                st.dataframe(pd.DataFrame(relatorio, columns=['Coluna', 'Método', 'Resultado']),
                             width='stretch')

# ============ TAB 3: MODELAGEM ============
with tabs[2]:
    st.header("🎯 Configuração e Execução")

    if not st.session_state.get('encoding_ready'):
        st.warning("⚠️ Aplique a codificação de categóricos na aba anterior.")
        st.stop()

    df_model = st.session_state['df_modeling']
    all_cols = list(df_model.columns)

    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("🎯 Variável Alvo", all_cols,
                              index=len(all_cols)-1, key='target_sel')
    with c2:
        features = st.multiselect("📊 Variáveis Preditoras",
                                  [c for c in all_cols if c != target],
                                  default=[c for c in all_cols if c != target][:10])

    if not features:
        st.warning("Selecione pelo menos uma variável preditora.")
        st.stop()

    if st.session_state['task_type'] == 'regression':
        st.subheader("⚙️ Parâmetros de Regressão")
        c1, c2, c3 = st.columns(3)
        with c1:
            outlier_mode = st.selectbox("Tratamento de outliers",
                ['remove_train', 'keep', 'winsorize', 'median'],
                format_func=lambda x: {'remove_train':'Remover do treino',
                                       'keep':'Manter','winsorize':'Winsorizar',
                                       'median':'Substituir por mediana'}[x])
        with c2:
            threshold = st.number_input("Threshold correlação", 0.0, 1.0, 0.10, 0.05)
        with c3:
            max_feat = st.number_input("Máx. features (0=todas)", 0, 50, 10)

        st.markdown("**Modelos:**")
        c1, c2, c3 = st.columns(3)
        use_lin = c1.checkbox("Regressão Linear", value=True)
        use_svr_lin = c2.checkbox("SVR Linear", value=True)
        use_svr_rbf = c3.checkbox("SVR RBF", value=True)

        st.markdown("**Grid Search:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("SVR Linear")
            C_lin = st.text_input("C (lin)", "0.1, 1, 10", key='C_lin')
            eps_lin = st.text_input("epsilon (lin)", "0.01, 0.1, 0.5", key='eps_lin')
        with c2:
            st.caption("SVR RBF")
            C_rbf = st.text_input("C (rbf)", "0.1, 1, 10", key='C_rbf')
            gamma_rbf = st.text_input("gamma (rbf)", "scale, auto, 0.1, 1", key='gamma_rbf')
            eps_rbf = st.text_input("epsilon (rbf)", "0.01, 0.1, 0.5", key='eps_rbf')
        with c3:
            search_method = st.selectbox("Método", ['grid','random','manual'])
            n_iter = st.number_input("n_iter (random)", 5, 500, 20)
            rs_choice = st.radio("Semente", ['fixa','aleatoria','custom'], horizontal=True)
            rs_val = 42 if rs_choice=='fixa' else (None if rs_choice=='aleatoria'
                     else st.number_input("Valor", 0, 9999, 42))

        def parse_list(txt):
            out = []
            for tok in str(txt).split(','):
                tok = tok.strip()
                if not tok: continue
                if tok.lower() in ('scale','auto'): out.append(tok.lower())
                else:
                    try: out.append(float(tok) if '.' in tok or 'e' in tok.lower() else int(tok))
                    except ValueError: out.append(tok)
            return out or [1]

        if st.button("▶️ Executar Análise de Regressão", width='stretch', type='primary'):
            grids = {'C_lin': parse_list(C_lin), 'eps_lin': parse_list(eps_lin),
                     'C_rbf': parse_list(C_rbf), 'gamma_rbf': parse_list(gamma_rbf),
                     'eps_rbf': parse_list(eps_rbf)}
            models = build_regression_models(grids, use_lin, use_svr_lin, use_svr_rbf)
            if not models:
                st.error("Selecione pelo menos um modelo."); st.stop()

            df_sub = df_model[features + [target]].dropna(subset=[target])
            X = df_sub[features]; y = df_sub[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                       random_state=rs_val or 42)
            X_tr_f, y_tr_f, removed = training_outlier_filter(X_tr, y_tr, outlier_mode)
            m = y_tr_f.notna()
            if (~m).any(): X_tr_f = X_tr_f.loc[m]; y_tr_f = y_tr_f.loc[m]

            results = evaluate_regression(
                X_tr_f, y_tr_f, X_te, y_te, models, threshold,
                max_feat if max_feat > 0 else None,
                cv=5, search_method=search_method, n_iter=n_iter,
                random_state=rs_val or 42)
            st.session_state['results'] = {
                'task': 'regression', 'results': results,
                'y_test': y_te, 'features': features, 'target': target,
                'removed': removed, 'train_n': len(X_tr_f), 'test_n': len(X_te)
            }
            st.success("🎉 Análise concluída! Veja a aba **Resultados**.")

    else:
        st.subheader("⚙️ Parâmetros de Classificação")
        n_unique = df_model[target].nunique()
        st.info(f"Alvo **{target}** tem **{n_unique}** classes únicas.")
        if n_unique > 20:
            st.error("⚠️ Ideal ≤ 20 classes. Considere discretizar ou usar regressão.")

        c1, c2 = st.columns(2)
        with c1:
            top_n = st.number_input("Avaliar top N classificadores", 5, 100, 20)
        with c2:
            cv_clf = st.number_input("Folds da CV", 3, 10, 5)

        all_classifiers = get_all_classifiers()
        st.write(f"**{len(all_classifiers)} classificadores disponíveis no scikit-learn.**")

        chosen = st.multiselect("Selecione os classificadores para avaliar",
                                all_classifiers,
                                default=all_classifiers[:min(top_n, len(all_classifiers))])

        if st.button("▶️ Executar Classificação", width='stretch', type='primary'):
            if not chosen:
                st.error("Selecione pelo menos um classificador."); st.stop()
            df_sub = df_model[features + [target]].dropna(subset=[target])
            le = LabelEncoder()
            y_enc = le.fit_transform(df_sub[target].astype(str))
            X = df_sub[features]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2,
                                                       random_state=42, stratify=y_enc)
            results = evaluate_classifiers(X_tr, y_tr, X_te, y_te, chosen,
                                            cv=cv_clf, top_n=len(chosen))
            st.session_state['results'] = {
                'task': 'classification', 'results': results,
                'y_test': y_te, 'features': features, 'target': target,
                'classes': list(le.classes_)
            }
            st.success("🎉 Classificação concluída! Veja a aba **Resultados**.")

# ============ TAB 4: RESULTADOS ============
with tabs[3]:
    st.header("📈 Resultados e Conclusões")

    if st.session_state.get('results') is None:
        st.info("Execute uma análise na aba **Modelagem** para ver os resultados.")
        st.stop()

    r = st.session_state['results']

    # ================= REGRESSÃO =================
    if r['task'] == 'regression':
        results = r['results']
        valid = {k: v for k, v in results.items() if 'error' not in v}

        rows = []
        for name, res in valid.items():
            params_txt = ', '.join(f'{k.replace("regressor__","")}={v}'
                                    for k, v in res['best_params'].items()) or '—'
            rows.append({
                'Modelo': name, 'CV R² médio': round(res['cv_mean'], 4),
                'CV R² desvio': round(res['cv_std'], 4),
                'R² teste': round(res['test_r2'], 4),
                'RMSE teste': round(res['test_rmse'], 4),
                'MAE teste': round(res['test_mae'], 4),
                'Hiperparâmetros': params_txt
            })
        df_tab = pd.DataFrame(rows).sort_values('CV R² médio', ascending=False)
        st.subheader("📋 Tabela Comparativa")
        st.dataframe(df_tab, width='stretch')

        best_cv = df_tab.iloc[0]['Modelo']
        st.success(f"🏆 **Melhor modelo (R² médio na CV):** {best_cv}")

        st.subheader("📊 Gráficos Comparativos")
        y_test = r['y_test']

        # Real vs Predito + Resíduos (matplotlib)
        c1, c2 = st.columns(2)
        with c1:
            fig, axes = plt.subplots(len(valid), 2, figsize=(10, 4*len(valid)))
            if len(valid) == 1: axes = np.array([axes])
            for i, (name, res) in enumerate(valid.items()):
                pred = res['y_pred']
                axes[i,0].scatter(y_test, pred, alpha=0.6, color='steelblue')
                lo = min(np.min(y_test), np.min(pred)); hi = max(np.max(y_test), np.max(pred))
                axes[i,0].plot([lo,hi],[lo,hi],'r--',lw=2)
                axes[i,0].set_xlabel('Real'); axes[i,0].set_ylabel('Predito')
                axes[i,0].set_title(f'{name} — Real vs Predito', fontsize=9)
                resid = np.asarray(y_test) - np.asarray(pred)
                axes[i,1].scatter(pred, resid, alpha=0.6, color='coral')
                axes[i,1].axhline(0, color='r', linestyle='--')
                axes[i,1].set_xlabel('Predito'); axes[i,1].set_ylabel('Resíduo')
                axes[i,1].set_title(f'{name} — Resíduos', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = list(valid.keys())
            x = np.arange(len(names)); w = 0.35
            ax.bar(x - w/2, [valid[n]['cv_mean'] for n in names], w, label='CV R²', color='#1e3c72')
            ax.bar(x + w/2, [valid[n]['test_r2'] for n in names], w, label='Teste R²', color='#2a5298')
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
            ax.set_ylabel('R²'); ax.set_title('Comparação CV × Teste')
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ============ BOXPLOT INTERATIVO (PLOTLY) ============
        st.subheader("📦 Boxplot Interativo — R² por Dobra da CV")
        st.caption("💡 Passe o mouse para ver detalhes, dê zoom, ou clique na legenda para isolar um modelo.")
        fig_box, box_df = plotly_boxplot_regression(valid)
        st.plotly_chart(fig_box, width='stretch')

        # Tabela detalhada por dobra
        with st.expander("🔍 Ver dados do boxplot (R² por dobra)"):
            st.dataframe(box_df.pivot_table(index='Modelo', columns='Fold',
                                             values='R²').round(4),
                         width='stretch')

    # ================= CLASSIFICAÇÃO =================
    else:
        results = r['results']
        valid = {k: v for k, v in results.items() if 'error' not in v}

        rows = []
        for name, res in valid.items():
            rows.append({
                'Classificador': name,
                'CV Acurácia': f"{res['cv_acc_mean']:.4f} ± {res['cv_acc_std']:.4f}",
                'Acurácia Teste': round(res['test_accuracy'], 4),
                'Precisão': round(res['test_precision'], 4),
                'Recall': round(res['test_recall'], 4),
                'F1-Score': round(res['test_f1'], 4)
            })
        df_tab = pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
        st.subheader("📋 Tabela Comparativa")
        st.dataframe(df_tab, width='stretch', height=400)

        if not df_tab.empty:
            best_name = df_tab.iloc[0]['Classificador']
            st.success(f"🏆 **Melhor classificador (F1-Score):** {best_name}")

        # Gráfico de barras
        if valid:
            fig, ax = plt.subplots(figsize=(12, 6))
            names = list(valid.keys())[:15]
            x = np.arange(len(names)); w = 0.2
            for i, (met, key) in enumerate([('Acurácia','test_accuracy'),
                                             ('Precisão','test_precision'),
                                             ('Recall','test_recall'),
                                             ('F1','test_f1')]):
                ax.bar(x + i*w, [valid[n][key] for n in names], w, label=met)
            ax.set_xticks(x + 1.5*w)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Score'); ax.set_title('Top classificadores — Métricas no Teste')
            ax.legend(); ax.set_ylim(0, 1.05)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ============ BOXPLOTS INTERATIVOS (PLOTLY) ============
        st.subheader("📦 Boxplots Interativos — Classificação")
        st.caption("💡 Gráficos interativos: hover, zoom e filtro pela legenda.")

        fig_box1, fig_box2, metr_df, fold_df = plotly_boxplot_classification(valid)

        st.markdown("**1) Distribuição das métricas entre os classificadores**")
        st.plotly_chart(fig_box1, width='stretch')

        if fig_box2 is not None:
            st.markdown("**2) Acurácia por dobra da CV — Top 15 classificadores**")
            st.plotly_chart(fig_box2, width='stretch')

            with st.expander("🔍 Ver dados de acurácia por dobra"):
                pivot = fold_df.pivot_table(index='Classificador', columns='Fold',
                                             values='Acurácia').round(4)
                st.dataframe(pivot, width='stretch')

        with st.expander("🔍 Ver dados das métricas por classificador"):
            st.dataframe(metr_df.pivot_table(index='Modelo', columns='Métrica',
                                              values='Valor').round(4),
                         width='stretch')

    # ================= EXPORTAÇÃO =================
    st.divider()
    st.subheader("💾 Exportar Resultados")
    c1, c2 = st.columns(2)
    with c1:
        csv = df_tab.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar resultados (.csv)", csv,
                            file_name=f"resultados_{datetime.now():%Y%m%d_%H%M%S}.csv",
                            mime='text/csv', width='stretch')
    with c2:
        export = {'tarefa': r['task'], 'alvo': r['target'],
                  'features': r['features'],
                  'modelos': {k: {kk: vv for kk, vv in v.items()
                                   if kk not in ('best_model','y_pred','cv_scores',
                                                  'grid_table','cv_acc_scores','cv_f1_scores')}
                               for k, v in results.items() if 'error' not in v}}
        js = json.dumps(export, indent=2, default=str).encode('utf-8')
        st.download_button("📥 Baixar relatório (.json)", js,
                            file_name=f"relatorio_{datetime.now():%Y%m%d_%H%M%S}.json",
                            mime='application/json', width='stretch')

# ---------- Rodapé ----------
st.divider()
st.caption("🧠 **Análise de Inteligência Computacional** — Interface Streamlit + Plotly | "
           "Regressão e Classificação com validação cruzada.")