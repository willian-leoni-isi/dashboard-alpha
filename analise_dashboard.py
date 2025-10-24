import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
import math

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Composição de Câmeras")
st.title("🎥 Simulador de Composição de Câmeras - AlphaPose")
st.markdown("**Encontre a combinação mínima de câmeras para cobertura máxima de keypoints**")
st.caption("📹 16 câmeras × 450 frames × 26 keypoints | 15 segundos de ação em múltiplas perspectivas")

# --- MAPEAMENTO DE ÂNGULOS DAS CÂMERAS ---
CAMERA_ANGLES = {
    'cam_01': 0.0,      # 12h
    'cam_02': 22.5,
    'cam_03': 45.0,
    'cam_04': 67.5,
    'cam_05': 90.0,     # 3h
    'cam_06': 112.5,
    'cam_07': 135.0,
    'cam_08': 157.5,
    'cam_09': 180.0,    # 6h
    'cam_10': 202.5,
    'cam_11': 225.0,
    'cam_12': 247.5,
    'cam_13': 270.0,    # 9h
    'cam_14': 292.5,
    'cam_15': 315.0,
    'cam_16': 337.5
}

ANGULO_MINIMO = 45.0  # Ângulo mínimo entre câmeras

def calcular_distancia_angular(angulo1, angulo2):
    """Calcula a menor distância angular entre dois ângulos (0-360°)"""
    diff = abs(angulo1 - angulo2)
    return min(diff, 360 - diff)

def validar_angulo_minimo(cameras_selecionadas):
    """Verifica se todas as câmeras respeitam o ângulo mínimo"""
    if len(cameras_selecionadas) < 2:
        return True, []
    
    violacoes = []
    cameras_ordenadas = sorted(cameras_selecionadas, key=lambda x: CAMERA_ANGLES[x])
    
    for i in range(len(cameras_ordenadas)):
        for j in range(i + 1, len(cameras_ordenadas)):
            cam1, cam2 = cameras_ordenadas[i], cameras_ordenadas[j]
            dist = calcular_distancia_angular(CAMERA_ANGLES[cam1], CAMERA_ANGLES[cam2])
            if dist < ANGULO_MINIMO:
                violacoes.append({
                    'cam1': cam1,
                    'cam2': cam2,
                    'angulo': dist
                })
    
    return len(violacoes) == 0, violacoes

# --- MAPEAMENTO DE KEYPOINTS PARA PORTUGUÊS (Halpe 26 Format) ---
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
    'right_ankle': 'Tornozelo Direito',
    'head': 'Cabeça',
    'neck': 'Pescoço',
    'hip': 'Centro do Quadril',
    'left_big_toe': 'Dedão Esquerdo',
    'right_big_toe': 'Dedão Direito',
    'left_small_toe': 'Dedinho Esquerdo',
    'right_small_toe': 'Dedinho Direito',
    'left_heel': 'Calcanhar Esquerdo',
    'right_heel': 'Calcanhar Direito'
}

# --- CARREGAMENTO DOS DADOS ---
@st.cache_data
def carregar_dados(caminho_excel):
    """Carrega os dados do arquivo Excel com offset corrigido."""
    try:
        df = pd.read_excel(caminho_excel)
        df['keypoint_name_pt'] = df['keypoint_name'].map(KEYPOINT_MAPPING)
        df['keypoint_name_pt'].fillna(df['keypoint_name'], inplace=True)
        st.toast("✅ Dados carregados com sucesso!", icon="📊")
        return df
    except FileNotFoundError:
        st.error(f"❌ Arquivo não encontrado: '{caminho_excel}'")
        # MELHORIA DE LOG: st.warning é mais apropriado que st.info
        st.warning("Execute primeiro: `python json_processor.py` para gerar o arquivo.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
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
    cobertura_geral = (total_cobertos / total_possibilidades) * 100 if total_possibilidades > 0 else 0
    
    # Cobertura por keypoint
    cobertura_por_kp = {}
    for kp in df['keypoint_name_pt'].unique():
        frames_kp_total = total_frames
        frames_kp_cobertos = df_filtrado[df_filtrado['keypoint_name_pt'] == kp]['frame_id'].nunique()
        cobertura_por_kp[kp] = (frames_kp_cobertos / frames_kp_total) * 100 if frames_kp_total > 0 else 0
    
    return cobertura_geral, cobertura_por_kp

# --- FUNÇÃO PARA ENCONTRAR MELHOR COMBINAÇÃO ---
@st.cache_data
def encontrar_melhores_combinacoes(df, limiar_confianca, max_cameras=6, respeitar_angulo=True):
    """Encontra as melhores combinações de 1 a max_cameras câmeras."""
    cameras_disponiveis = sorted(df['camera_id'].unique())
    resultados = []
    
    progress_bar = st.progress(0, text="Iniciando análise de combinações...")
    status_text = st.empty()
    
    total_combinacoes_possiveis = sum(len(list(combinations(cameras_disponiveis, n))) for n in range(1, min(max_cameras + 1, len(cameras_disponiveis) + 1)))
    
    # MELHORIA DE LOG: Informa o usuário antes do loop começar
    status_text.info(f"Preparando {total_combinacoes_possiveis:,} combinações... (pode levar um momento)")
    
    resultados_validos = []
    contador_total = 0
    
    for n in range(1, min(max_cameras + 1, len(cameras_disponiveis) + 1)):
        combos = list(combinations(cameras_disponiveis, n))
        for combo in combos:
            contador_total += 1
            # Valida ângulo mínimo se necessário
            if respeitar_angulo:
                valido, _ = validar_angulo_minimo(list(combo))
                if not valido:
                    continue # Pula esta combinação, não a adiciona aos resultados
            
            cobertura, _ = calcular_cobertura(df, list(combo), limiar_confianca)
            resultados_validos.append({
                'num_cameras': n,
                'cameras': ', '.join(combo),
                'cobertura': cobertura
            })
            
            if contador_total % 100 == 0:
                progresso = min(contador_total / total_combinacoes_possiveis, 1.0)
                progress_bar.progress(progresso, text=f"Analisando combinação {contador_total:,}/{total_combinacoes_possiveis:,}...")
                status_text.empty() # Limpa a mensagem "Preparando..."

    progress_bar.empty()
    status_text.empty()
    
    if not resultados_validos:
        st.warning("Nenhuma combinação válida encontrada com os filtros aplicados.")
        return pd.DataFrame(columns=['num_cameras', 'cameras', 'cobertura'])

    return pd.DataFrame(resultados_validos).sort_values('cobertura', ascending=False)

# --- FUNÇÃO PARA CRIAR DIAGRAMA CIRCULAR ---
def criar_diagrama_circular(cameras_selecionadas, violacoes=[]):
    """Cria um diagrama circular mostrando as câmeras selecionadas"""
    fig = go.Figure()
    
    # Círculo externo
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Linhas cardeais
    for angulo, cor, nome in [(0, 'lightblue', 'Norte'), (90, 'lightgreen', 'Leste'), 
                               (180, 'lightcoral', 'Sul'), (270, 'lightyellow', 'Oeste')]:
        rad = math.radians(angulo - 90)
        fig.add_trace(go.Scatter(
            x=[0, 1.15*math.cos(rad)],
            y=[0, 1.15*math.sin(rad)],
            mode='lines',
            line=dict(color=cor, width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Adiciona violações (linhas vermelhas)
    for v in violacoes:
        ang1 = math.radians(CAMERA_ANGLES[v['cam1']] - 90)
        ang2 = math.radians(CAMERA_ANGLES[v['cam2']] - 90)
        fig.add_trace(go.Scatter(
            x=[0.9*math.cos(ang1), 0.9*math.cos(ang2)],
            y=[0.9*math.sin(ang1), 0.9*math.sin(ang2)],
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Violação {v["angulo"]:.1f}°',
            hovertext=f'{v["cam1"]} ↔ {v["cam2"]}: {v["angulo"]:.1f}°'
        ))
    
    # Adiciona câmeras
    for cam, angulo in CAMERA_ANGLES.items():
        rad = math.radians(angulo - 90)  # -90 para começar no topo
        x = math.cos(rad)
        y = math.sin(rad)
        
        selecionada = cam in cameras_selecionadas
        cor = '#10b981' if selecionada else '#d1d5db'
        tamanho = 20 if selecionada else 12
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=tamanho, color=cor, line=dict(color='white', width=2)),
            text=cam.replace('cam_', 'C'),
            textposition='middle center',
            textfont=dict(size=8, color='white' if selecionada else 'gray'),
            name=f'{cam} ({angulo}°)',
            hovertext=f'{cam}<br>Ângulo: {angulo}°<br>Status: {"✅ Selecionada" if selecionada else "⚪ Disponível"}'
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor='x'),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"Disposição Circular das Câmeras (Ângulo mínimo: {ANGULO_MINIMO}°)",
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig

# --- CARREGAMENTO DOS DADOS ---
df = carregar_dados('dados_processados_por_camera.xlsx')

if df is not None:
    cameras_disponiveis = sorted(df['camera_id'].unique())
    
    # Informações do dataset
    st.sidebar.markdown(f"""
    ### 📊 Informações do Dataset
    - **Câmeras:** {len(cameras_disponiveis)} perspectivas
    - **Frames:** {df['frame_id'].nunique():,} (15 segundos)
    - **Keypoints:** {df['keypoint_name_pt'].nunique()}
    - **Detecções:** {len(df):,}
    - **Ângulo entre câmeras:** 22.5°
    """)
    
    st.sidebar.markdown("---")
    
    # --- SIDEBAR: CONTROLES ---
    st.sidebar.header("⚙️ Controles do Simulador")
    
    # Controle 1: Limiar de Confiança
    st.sidebar.subheader("1. Limiar de Confiança")
    limiar_confianca = st.sidebar.slider(
        "Confiança mínima (%)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Detecções abaixo deste valor serão ignoradas"
    )
    
    # Controle 2: Seleção de Câmeras
    st.sidebar.subheader("2. Selecione as Câmeras")
    
    st.sidebar.info(f"""
    **⚠️ Regra de Ângulo Mínimo**
    
    As câmeras devem ter pelo menos **{ANGULO_MINIMO}°** de separação entre si para garantir perspectivas distintas.
    
    **Posições Cardeais:**
    - 🔵 Norte (12h): C1 (0°)
    - 🟢 Leste (3h): C5 (90°)  
    - 🔴 Sul (6h): C9 (180°)
    - 🟡 Oeste (9h): C13 (270°)
    """)
    
    # Opção de seleção rápida
    selecao_rapida = st.sidebar.selectbox(
        "Seleção Rápida:",
        [
            "Manual",
            "Todas (16)",
            "Nenhuma", 
            "Cardeais (C1,C5,C9,C13)",
            "Diagonais (C3,C7,C11,C15)",
            "8 Principais (45° entre si)",
            "Metade Norte (C16,C1-C8)",
            "Metade Sul (C9-C16)"
        ],
        index=0
    )
    
    # A lista 'cameras_selecionadas' será a fonte da verdade para os checkboxes
    # Esta lógica é executada ANTES dos checkboxes serem desenhados
    cameras_selecionadas_preset = []
    
    if selecao_rapida == "Todas (16)":
        cameras_selecionadas_preset = cameras_disponiveis.copy()
    elif selecao_rapida == "Cardeais (C1,C5,C9,C13)":
        cameras_selecionadas_preset = ['cam_01', 'cam_05', 'cam_09', 'cam_13']
    elif selecao_rapida == "Diagonais (C3,C7,C11,C15)":
        cameras_selecionadas_preset = ['cam_03', 'cam_07', 'cam_11', 'cam_15']
    elif selecao_rapida == "8 Principais (45° entre si)":
        cameras_selecionadas_preset = ['cam_01', 'cam_03', 'cam_05', 'cam_07', 
                                'cam_09', 'cam_11', 'cam_13', 'cam_15']
    elif selecao_rapida == "Metade Norte (C16,C1-C8)":
        cameras_selecionadas_preset = ['cam_16', 'cam_01', 'cam_02', 'cam_03', 
                                'cam_04', 'cam_05', 'cam_06', 'cam_07', 'cam_08']
    elif selecao_rapida == "Metade Sul (C9-C16)":
        # CORREÇÃO DE BUG: Faltava a 'cam_16'
        cameras_selecionadas_preset = ['cam_09', 'cam_10', 'cam_11', 'cam_12',
                                'cam_13', 'cam_14', 'cam_15', 'cam_16']
    # Se for "Manual" ou "Nenhuma", a lista fica vazia (cameras_selecionadas_preset = [])
    
    
    st.sidebar.markdown("**Seleção Individual:**")
    
    # A lista final será (re)construída a partir do estado dos checkboxes
    cameras_selecionadas = []
    
    # Divide em 4 colunas para melhor visualização das 16 câmeras
    cols = st.sidebar.columns(4)
    for idx, cam in enumerate(cameras_disponiveis):
        col = cols[idx % 4]
        
        # CORREÇÃO LÓGICA: 
        # A seleção "Manual" (index 0) não deve mais selecionar as 8 primeiras por padrão.
        # Agora, o 'value' do checkbox é determinado *apenas* pela lista 'cameras_selecionadas_preset'
        # que é controlada pela 'selecao_rapida'.
        checked = cam in cameras_selecionadas_preset
        
        label = f"{cam.replace('cam_', 'C')} ({CAMERA_ANGLES[cam]:.0f}°)"
        
        # O estado 'value=checked' define o checkbox. 
        # A interação do usuário é capturada pelo 'if col.checkbox(...)'
        if col.checkbox(label, value=checked, key=f"cam_{cam}"):
            if cam not in cameras_selecionadas:
                cameras_selecionadas.append(cam)
        else:
            # Esta parte não é estritamente necessária se reconstruímos a lista do zero,
            # mas é boa prática para o 'if/else'
            pass 
            
    # Validação de ângulo
    angulo_valido, violacoes = validar_angulo_minimo(cameras_selecionadas)
    
    if not angulo_valido:
        st.sidebar.error(f"""
        ⚠️ **Violação de Ângulo Mínimo!**
        
        {len(violacoes)} par(es) de câmeras estão muito próximos (<{ANGULO_MINIMO}°):
        """)
        for v in violacoes[:5]:  # Mostra no máximo 5
            st.sidebar.write(f"• {v['cam1']} ↔ {v['cam2']}: {v['angulo']:.1f}°")
        if len(violacoes) > 5:
            st.sidebar.write(f"... e mais {len(violacoes)-5} violações")
    else:
        if len(cameras_selecionadas) > 0:
            # MELHORIA DE LOG: Mensagem mais clara
            st.sidebar.success(f"✅ Configuração válida! {len(cameras_selecionadas)} câmeras selecionadas.")
        else:
            st.sidebar.warning("Nenhuma câmera selecionada.")
    
    # --- DIAGRAMA CIRCULAR ---
    st.header("🎯 Disposição Espacial das Câmeras")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_circular = criar_diagrama_circular(cameras_selecionadas, violacoes if not angulo_valido else [])
        st.plotly_chart(fig_circular, use_container_width=True)
    
    with col2:
        st.markdown("### Legenda")
        st.markdown("""
        - 🟢 **Verde**: Câmera selecionada
        - ⚪ **Cinza**: Câmera disponível  
        - 🔴 **Linha vermelha**: Violação de ângulo
        
        **Dica:** Use a "Seleção Rápida" ou clique nas câmeras na sidebar para montar sua composição.
        """)
        
        if cameras_selecionadas:
            st.markdown("### Câmeras Ativas")
            for cam in sorted(cameras_selecionadas, key=lambda x: CAMERA_ANGLES[x]):
                st.markdown(f"• **{cam}** → {CAMERA_ANGLES[cam]}°")
    
    # --- CÁLCULO DA COBERTURA ---
    if cameras_selecionadas and angulo_valido:
        cobertura_geral, cobertura_por_kp = calcular_cobertura(df, cameras_selecionadas, limiar_confianca)
    else:
        cobertura_geral = 0.0
        cobertura_por_kp = {kp: 0.0 for kp in df['keypoint_name_pt'].unique()}
    
    # --- PLACAR PRINCIPAL ---
    st.header("📊 Pontuação de Cobertura da Composição")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Placar grande
        if not angulo_valido:
            st.error("⚠️ Configuração inválida - Corrija as violações de ângulo primeiro")
        elif not cameras_selecionadas:
            st.info("Selecione uma ou mais câmeras para calcular a cobertura.")
        else:
            cor_placar = "🟢" if cobertura_geral >= 90 else "🟡" if cobertura_geral >= 70 else "🔴"
            st.markdown(f"### {cor_placar} Cobertura Total: **{cobertura_geral:.1f}%**")
            st.progress(cobertura_geral / 100)
            st.caption(f"Limiar: {limiar_confianca}% | Câmeras: {len(cameras_selecionadas)}/{len(cameras_disponiveis)}")
    
    with col2:
        # Frames cobertos
        if cameras_selecionadas and angulo_valido:
            df_filtrado = df[df['camera_id'].isin(cameras_selecionadas)]
            df_filtrado = df_filtrado[df_filtrado['confidence'] >= limiar_confianca / 100.0]
            frames_cobertos = df_filtrado['frame_id'].nunique()
        else:
            frames_cobertos = 0
        total_frames = df['frame_id'].nunique()
        st.metric("Frames Cobertos", f"{frames_cobertos}/{total_frames}")
    
    with col3:
        # Keypoints problemáticos
        kps_ruins = sum(1 for v in cobertura_por_kp.values() if v < 70)
        delta_color = "normal" if kps_ruins == 0 else "inverse"
        st.metric("Pontos Cegos", kps_ruins, 
                 delta=None if kps_ruins == 0 else f"{kps_ruins} KPs < 70%",
                 delta_color=delta_color)
    
    # --- GRÁFICO DE PONTOS CEGOS ---
    if angulo_valido and cameras_selecionadas:
        st.header("📉 Análise de Pontos Cegos da Composição")
        
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
            height=700,
            showlegend=False,
            xaxis_range=[0, 105] # Garante espaço para o texto
        )
        fig_pontos_cegos.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Limiar Aceitável")
        fig_pontos_cegos.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Excelente")
        
        st.plotly_chart(fig_pontos_cegos, use_container_width=True)
    
    # --- RECOMENDAÇÕES AUTOMÁTICAS ---
    st.header("💡 Recomendações de Combinações Ótimas")
    
    respeitar_angulo_busca = st.checkbox(
        f"Considerar apenas combinações com ângulo mínimo de {ANGULO_MINIMO}°",
        value=True,
        help="Se marcado, apenas combinações que respeitam o ângulo mínimo serão consideradas"
    )
    
    if st.button("Encontrar Melhores Combinações (até 6 câmeras)"):
        # O @st.cache_data vai garantir que isso só rode se os parâmetros mudarem
        df_combinacoes = encontrar_melhores_combinacoes(
            df, 
            limiar_confianca, 
            max_cameras=6, 
            respeitar_angulo=respeitar_angulo_busca
        )
        
        if df_combinacoes.empty:
            st.warning("Nenhuma combinação encontrada. Tente desmarcar a restrição de ângulo.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 10 Melhores Combinações")
                top_10 = df_combinacoes.head(10).copy()
                top_10['cobertura'] = top_10['cobertura'].round(1)
                st.dataframe(
                    top_10,
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
                st.subheader("📊 Melhor por Quantidade")
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
            st.header("💰 Análise de Custo-Benefício")
            
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

else:
    st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo 'dados_processados_por_camera.xlsx' existe no diretório.")