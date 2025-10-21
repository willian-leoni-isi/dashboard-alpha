import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Composição de Câmeras")
st.title("Simulador de Composição de Câmeras - AlphaPose")
st.markdown("**Encontre a combinação mínima de câmeras para cobertura máxima de keypoints**")

# --- MAPEAMENTO DE KEYPOINTS PARA PORTUGUÊS ---
KEYPOINT_MAPPING = {
    'nose': 'Nariz',
    'left_eye': 'Olho Esquerdo',
    'right_eye': 'Olho Direito',
    'left_ear': 'Orelha Esquerda',
    'right_ear': 'Orelha Direita',
    'left_shoulder': 'Ombro Esquerdo',
    'right_shoulder': 'Ombro Direito',
    'left_elbow': 'Cotovelo Esquerdo',
    'right_elbow': 'Cotovelo Direito',
    'left_wrist': 'Pulso Esquerdo',
    'right_wrist': 'Pulso Direito',
    'left_hip': 'Quadril Esquerdo',
    'right_hip': 'Quadril Direito',
    'left_knee': 'Joelho Esquerdo',
    'right_knee': 'Joelho Direito',
    'left_ankle': 'Tornozelo Esquerdo',
    'right_ankle': 'Tornozelo Direito'
}

# --- CARREGAMENTO DOS DADOS ---
@st.cache_data
def carregar_dados(caminho_excel):
    """Carrega os dados do arquivo Excel com offset corrigido."""
    try:
        df = pd.read_excel(caminho_excel)
        df['keypoint_name_pt'] = df['keypoint_name'].map(KEYPOINT_MAPPING)
        df['keypoint_name_pt'].fillna(df['keypoint_name'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: '{caminho_excel}'")
        return None

# --- FUNÇÃO DE CÁLCULO DE COBERTURA ---
def calcular_cobertura(df, cameras_selecionadas, limiar_confianca):
    """
    Calcula a cobertura de keypoints para um conjunto de câmeras.
    
    Cobertura = % de (frame, keypoint) cobertos acima do limiar por pelo menos 1 câmera.
    """
    if not cameras_selecionadas:
        return 0.0, {}
    
    # Filtra apenas as câmeras selecionadas
    df_filtrado = df[df['camera_id'].isin(cameras_selecionadas)].copy()
    
    # Considera apenas detecções acima do limiar
    df_filtrado = df_filtrado[df_filtrado['confidence'] >= limiar_confianca / 100.0]
    
    # Total de possibilidades: todos os frames x todos os keypoints
    total_frames = df['frame_id'].nunique()
    total_keypoints = df['keypoint_name_pt'].nunique()
    total_possibilidades = total_frames * total_keypoints
    
    # Cobertura: quantos (frame, keypoint) únicos foram cobertos
    cobertos = df_filtrado.groupby(['frame_id', 'keypoint_name_pt']).size()
    total_cobertos = len(cobertos)
    
    # Cobertura geral
    cobertura_geral = (total_cobertos / total_possibilidades) * 100
    
    # Cobertura por keypoint
    cobertura_por_kp = {}
    for kp in df['keypoint_name_pt'].unique():
        frames_kp_total = total_frames
        frames_kp_cobertos = df_filtrado[df_filtrado['keypoint_name_pt'] == kp]['frame_id'].nunique()
        cobertura_por_kp[kp] = (frames_kp_cobertos / frames_kp_total) * 100
    
    return cobertura_geral, cobertura_por_kp

# --- FUNÇÃO PARA ENCONTRAR MELHOR COMBINAÇÃO ---
@st.cache_data
def encontrar_melhores_combinacoes(df, limiar_confianca, max_cameras=4):
    """Encontra as melhores combinações de 1 a max_cameras câmeras."""
    cameras_disponiveis = sorted(df['camera_id'].unique())
    resultados = []
    
    for n in range(1, min(max_cameras + 1, len(cameras_disponiveis) + 1)):
        for combo in combinations(cameras_disponiveis, n):
            cobertura, _ = calcular_cobertura(df, list(combo), limiar_confianca)
            resultados.append({
                'num_cameras': n,
                'cameras': ', '.join(combo),
                'cobertura': cobertura
            })
    
    return pd.DataFrame(resultados).sort_values('cobertura', ascending=False)

# --- CARREGAMENTO DOS DADOS ---
df = carregar_dados('dados_processados_por_camera.xlsx')

if df is not None:
    cameras_disponiveis = sorted(df['camera_id'].unique())
    
    # --- SIDEBAR: CONTROLES ---
    st.sidebar.header("Controles do Simulador")
    
    # Controle 1: Limiar de Confiança
    st.sidebar.subheader("1. Limiar de Confiança")
    limiar_confianca = st.sidebar.slider(
        "Confiança mínima para considerar detecção válida (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Detecções abaixo deste valor serão ignoradas"
    )
    
    # Controle 2: Seleção de Câmeras
    st.sidebar.subheader("2. Selecione as Câmeras")
    cameras_selecionadas = []
    
    for cam in cameras_disponiveis:
        if st.sidebar.checkbox(cam, value=True, key=f"cam_{cam}"):
            cameras_selecionadas.append(cam)
    
    # Botões de ação
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Selecionar Todas"):
            st.rerun()
    with col2:
        if st.button("Limpar Todas"):
            st.rerun()
    
    # --- CÁLCULO DA COBERTURA ---
    if cameras_selecionadas:
        cobertura_geral, cobertura_por_kp = calcular_cobertura(df, cameras_selecionadas, limiar_confianca)
    else:
        cobertura_geral = 0.0
        cobertura_por_kp = {kp: 0.0 for kp in df['keypoint_name_pt'].unique()}
    
    # --- PLACAR PRINCIPAL ---
    st.header("Pontuação de Cobertura da Composição")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Placar grande
        cor_placar = "🟢" if cobertura_geral >= 90 else "🟡" if cobertura_geral >= 70 else "🔴"
        st.markdown(f"### {cor_placar} Cobertura Total: **{cobertura_geral:.1f}%**")
        st.progress(cobertura_geral / 100)
        st.caption(f"Limiar de confiança: {limiar_confianca}% | Câmeras ativas: {len(cameras_selecionadas)}")
    
    with col2:
        # Frames cobertos
        df_filtrado = df[df['camera_id'].isin(cameras_selecionadas)]
        df_filtrado = df_filtrado[df_filtrado['confidence'] >= limiar_confianca / 100.0]
        frames_cobertos = df_filtrado['frame_id'].nunique()
        total_frames = df['frame_id'].nunique()
        st.metric("Frames Cobertos", f"{frames_cobertos}/{total_frames}")
    
    with col3:
        # Keypoints problemáticos
        kps_ruins = sum(1 for v in cobertura_por_kp.values() if v < 70)
        st.metric("Pontos Cegos", kps_ruins, delta=None if kps_ruins == 0 else f"{kps_ruins} keypoints < 70%")
    
    # --- GRÁFICO DE PONTOS CEGOS ---
    st.header("Análise de Pontos Cegos da Composição")
    
    # Prepara dados para o gráfico
    df_cobertura_kp = pd.DataFrame([
        {'keypoint': k, 'cobertura': v} 
        for k, v in cobertura_por_kp.items()
    ]).sort_values('cobertura', ascending=True)
    
    # Define cores baseadas na cobertura
    cores = ['#ef4444' if x < 70 else '#f59e0b' if x < 90 else '#10b981' 
             for x in df_cobertura_kp['cobertura']]
    
    fig_pontos_cegos = go.Figure()
    fig_pontos_cegos.add_trace(go.Bar(
        y=df_cobertura_kp['keypoint'],
        x=df_cobertura_kp['cobertura'],
        orientation='h',
        marker=dict(color=cores),
        text=df_cobertura_kp['cobertura'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig_pontos_cegos.update_layout(
        title="Cobertura por Keypoint (ordenado do pior para o melhor)",
        xaxis_title="Cobertura (%)",
        yaxis_title="",
        height=600,
        showlegend=False
    )
    fig_pontos_cegos.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Limiar Aceitável (70%)")
    fig_pontos_cegos.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Excelente (90%)")
    
    st.plotly_chart(fig_pontos_cegos, use_container_width=True)
    
    # --- RECOMENDAÇÕES AUTOMÁTICAS ---
    st.header("Recomendações de Combinações Ótimas")
    
    with st.spinner("Calculando todas as combinações possíveis..."):
        df_combinacoes = encontrar_melhores_combinacoes(df, limiar_confianca, max_cameras=4)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Melhores Combinações")
        top_5 = df_combinacoes.head(5).copy()
        top_5['cobertura'] = top_5['cobertura'].round(1)
        st.dataframe(
            top_5,
            use_container_width=True,
            hide_index=True,
            column_config={
                'num_cameras': 'Nº Câmeras',
                'cameras': 'Combinação',
                'cobertura': st.column_config.ProgressColumn(
                    'Cobertura (%)',
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
    
    with col2:
        st.subheader("Melhor por Quantidade de Câmeras")
        melhor_por_qtd = df_combinacoes.loc[df_combinacoes.groupby('num_cameras')['cobertura'].idxmax()]
        melhor_por_qtd['cobertura'] = melhor_por_qtd['cobertura'].round(1)
        st.dataframe(
            melhor_por_qtd,
            use_container_width=True,
            hide_index=True,
            column_config={
                'num_cameras': 'Nº Câmeras',
                'cameras': 'Melhor Combinação',
                'cobertura': st.column_config.ProgressColumn(
                    'Cobertura (%)',
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
    
    # --- ANÁLISE DE CUSTO-BENEFÍCIO ---
    st.header("Análise de Custo-Benefício")
    
    # Gráfico mostrando ganho marginal
    melhor_por_qtd_sorted = melhor_por_qtd.sort_values('num_cameras')
    melhor_por_qtd_sorted['ganho_marginal'] = melhor_por_qtd_sorted['cobertura'].diff().fillna(melhor_por_qtd_sorted['cobertura'])
    
    fig_custo = go.Figure()
    fig_custo.add_trace(go.Scatter(
        x=melhor_por_qtd_sorted['num_cameras'],
        y=melhor_por_qtd_sorted['cobertura'],
        mode='lines+markers',
        name='Cobertura Total',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10)
    ))
    
    fig_custo.update_layout(
        title="Cobertura vs Número de Câmeras",
        xaxis_title="Número de Câmeras",
        yaxis_title="Cobertura (%)",
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    st.plotly_chart(fig_custo, use_container_width=True)
    
    st.caption("**Interpretação:** Procure o 'joelho' da curva - ponto onde adicionar mais câmeras traz ganhos marginais pequenos.")
    
    # --- TIMELINE DE COBERTURA ---
    st.header("Timeline de Cobertura")
    st.write("Visualize a cobertura ao longo do tempo (frames)")
    
    if cameras_selecionadas:
        # Calcula cobertura por frame
        df_filtrado = df[df['camera_id'].isin(cameras_selecionadas)]
        df_filtrado = df_filtrado[df_filtrado['confidence'] >= limiar_confianca / 100.0]
        
        cobertura_por_frame = []
        for frame in sorted(df['frame_id'].unique()):
            kps_no_frame = df_filtrado[df_filtrado['frame_id'] == frame]['keypoint_name_pt'].nunique()
            total_kps = df['keypoint_name_pt'].nunique()
            cobertura_frame = (kps_no_frame / total_kps) * 100
            cobertura_por_frame.append({
                'frame': int(frame),
                'cobertura': cobertura_frame
            })
        
        df_timeline = pd.DataFrame(cobertura_por_frame)
        
        # Amostragem para não sobrecarregar o gráfico (mostra 1 a cada 10 frames)
        df_timeline_sample = df_timeline[::10]
        
        fig_timeline = px.line(
            df_timeline_sample,
            x='frame',
            y='cobertura',
            title="Cobertura de Keypoints ao Longo dos Frames",
            labels={'frame': 'Frame', 'cobertura': 'Cobertura (%)'}
        )
        fig_timeline.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Mínimo Aceitável")
        fig_timeline.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Excelente")
        fig_timeline.update_layout(height=400)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Identifica frames críticos
        frames_criticos = df_timeline[df_timeline['cobertura'] < 70]
        if not frames_criticos.empty:
            st.warning(f"⚠️ **{len(frames_criticos)} frames críticos** detectados (cobertura < 70%)")
            st.caption(f"Frames problemáticos: {frames_criticos['frame'].tolist()[:10]}{'...' if len(frames_criticos) > 10 else ''}")
    
    # --- DADOS BRUTOS ---
    with st.expander("Ver Dados Brutos e Estatísticas"):
        st.subheader("Estatísticas por Câmera")
        stats = df.groupby('camera_id').agg({
            'frame_id': 'nunique',
            'confidence': ['mean', 'std', 'min', 'max']
        }).round(3)
        st.dataframe(stats, use_container_width=True)
        
        st.subheader("Amostra dos Dados")
        st.dataframe(df.head(100), use_container_width=True)