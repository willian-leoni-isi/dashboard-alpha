import json
import pandas as pd
import os

# --------------------------------------------------------------------------
# --- CONFIGURAÇÕES CORRETAS ---
# --------------------------------------------------------------------------
VIDEO_SECONDS = 15  # 15 segundos por câmera (260s / 16 câmeras ≈ 16.25, mas 450 frames = 15s)
FPS = 30
NUM_CAMERAS = 16  # 7200 frames ÷ 450 frames/câmera = 16 câmeras
FRAMES_PER_CAMERA = 450  # 15 segundos × 30 FPS
INPUT_JSON_PATH = 'alphapose-results.json'
OUTPUT_EXCEL_PATH = 'dados_processados_por_camera.xlsx'
FRAME_PADDING = 3  # Para formatar 000, 001, ..., 449

# --------------------------------------------------------------------------
# --- MAPEAMENTO CORRETO DOS 26 KEYPOINTS (Halpe Format) ---
# --------------------------------------------------------------------------
# Fonte: https://github.com/Fang-Haoshu/Halpe-FullBody
KEYPOINT_NAMES = [
    "nose",                    # 0
    "left_eye",               # 1
    "right_eye",              # 2
    "left_ear",               # 3
    "right_ear",              # 4
    "left_shoulder",          # 5
    "right_shoulder",         # 6
    "left_elbow",             # 7
    "right_elbow",            # 8
    "left_wrist",             # 9
    "right_wrist",            # 10
    "left_hip",               # 11
    "right_hip",              # 12
    "left_knee",              # 13
    "right_knee",             # 14
    "left_ankle",             # 15
    "right_ankle",            # 16
    "head",                   # 17
    "neck",                   # 18
    "hip",                    # 19 (centro dos quadris)
    "left_big_toe",           # 20
    "right_big_toe",          # 21
    "left_small_toe",         # 22
    "right_small_toe",        # 23
    "left_heel",              # 24
    "right_heel"              # 25
]

# --------------------------------------------------------------------------
# --- LÓGICA PRINCIPAL (CORRIGIDA COM OFFSET) ---
# --------------------------------------------------------------------------

def processar_dados_alphapose():
    """
    Lê o JSON, calcula o ID da câmera (offset) e o FRAME DE REFERÊNCIA (módulo)
    para comparar corretamente as perspectivas e salva em Excel.
    
    IMPORTANTE: O vídeo tem 7200 frames totais (260s × 30fps, dividido em 16 perspectivas)
    - Frames 0-449: cam_01 (mesma cena, perspectiva 1) - 15 segundos
    - Frames 450-899: cam_02 (mesma cena, perspectiva 2) - 15 segundos
    - Frames 900-1349: cam_03 (mesma cena, perspectiva 3) - 15 segundos
    - ... e assim por diante até cam_16
    
    O frame_id normalizado (0-449) representa o MESMO MOMENTO no tempo.
    Exemplo: frame 0, 450, 900, 1350... = mesmo instante, 16 ângulos diferentes
    """

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"ERRO: Arquivo de entrada '{INPUT_JSON_PATH}' não encontrado.")
        return

    print("=" * 80)
    print("PROCESSADOR DE DADOS ALPHAPOSE - MÚLTIPLAS CÂMERAS")
    print("=" * 80)
    print(f"\nConfiguração:")
    print(f"  • Câmeras: {NUM_CAMERAS}")
    print(f"  • Frames por câmera: {FRAMES_PER_CAMERA} ({VIDEO_SECONDS}s × {FPS} FPS)")
    print(f"  • Total de frames esperado: {FRAMES_PER_CAMERA * NUM_CAMERAS}")
    print(f"  • Total de keypoints por detecção: {len(KEYPOINT_NAMES)}")
    print("-" * 80)

    try:
        print(f"\nCarregando JSON de '{INPUT_JSON_PATH}'...")
        with open(INPUT_JSON_PATH, 'r') as f:
            detections = json.load(f)
        print(f"✅ Total de {len(detections)} detecções encontradas no JSON.")
    except Exception as e:
        print(f"❌ ERRO ao ler o arquivo JSON: {e}")
        return

    processed_data = []
    frames_ignorados = 0
    frames_processados = 0
    
    print("\n📊 Processando dados com offset de câmeras...")

    for idx, detection in enumerate(detections):
        if (idx + 1) % 1000 == 0:
            print(f"   Processando detecção {idx + 1}/{len(detections)}...")
            
        original_frame_id = detection.get('image_id', '0.jpg')
        
        try:
            # Extrai o número do frame do image_id
            base_name, extension = os.path.splitext(original_frame_id)
            frame_number = int(base_name)

            # --- CÁLCULO DO ÍNDICE DA CÂMERA (OFFSET) ---
            # Cada bloco de 450 frames = 1 câmera
            camera_index = frame_number // FRAMES_PER_CAMERA
            
            # --- FILTRA FRAMES INVÁLIDOS ---
            if camera_index >= NUM_CAMERAS:
                if frames_ignorados < 5:  # Mostra apenas os primeiros avisos
                    print(f"   ⚠️ Aviso: Ignorando frame '{original_frame_id}' (câmera {camera_index + 1} > {NUM_CAMERAS})")
                frames_ignorados += 1
                continue
                
            camera_id = f"cam_{camera_index + 1:02d}"

            # --- CALCULA O FRAME DE REFERÊNCIA (MOMENTO NO TEMPO) ---
            # O momento no tempo (0-449) é o MESMO para todas as câmeras
            # Frame 0, 450, 900... = todos representam o tempo t=0
            ref_frame_num = frame_number % FRAMES_PER_CAMERA
            ref_frame_id = f"{ref_frame_num:0{FRAME_PADDING}d}"

            keypoints = detection.get('keypoints', [])
            score = detection.get('score', 0.0)
            
            # Validação do array de keypoints
            expected_length = len(KEYPOINT_NAMES) * 3
            if len(keypoints) != expected_length:
                if frames_ignorados < 5:
                    print(f"   ⚠️ Aviso: Frame '{original_frame_id}' tem {len(keypoints)} valores, esperado {expected_length}. Pulando.")
                frames_ignorados += 1
                continue
            
            # Processa cada keypoint (x, y, confidence)
            for kp_idx in range(0, len(keypoints), 3):
                point_index = kp_idx // 3
                if point_index < len(KEYPOINT_NAMES):
                    keypoint_name = KEYPOINT_NAMES[point_index]
                    x = keypoints[kp_idx]
                    y = keypoints[kp_idx + 1]
                    confidence = keypoints[kp_idx + 2]
                    
                    processed_data.append({
                        "camera_id": camera_id,
                        "frame_id": ref_frame_id,
                        "original_frame_id": original_frame_id,
                        "keypoint_name": keypoint_name,
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                        "detection_score": score
                    })
            
            frames_processados += 1
            
        except (ValueError, TypeError) as e:
            if frames_ignorados < 5:
                print(f"   ⚠️ Aviso: Ignorando detecção com image_id inválido: {original_frame_id} ({e})")
            frames_ignorados += 1
            continue

    if frames_ignorados > 5:
        print(f"   ... e mais {frames_ignorados - 5} frames ignorados")

    df = pd.DataFrame(processed_data)

    if df.empty:
        print("\n❌ ERRO: Nenhum dado foi processado. Verifique o formato do JSON.")
        return None

    try:
        print(f"\n💾 Salvando dados em formato Excel em '{OUTPUT_EXCEL_PATH}'...")
        
        # Reordena colunas para maior clareza
        colunas_ordenadas = [
            "camera_id", "frame_id", "keypoint_name", 
            "confidence", "x", "y", "detection_score", "original_frame_id"
        ]
        df = df[colunas_ordenadas]
        
        # Ordena por câmera e frame para facilitar análise
        df = df.sort_values(['camera_id', 'frame_id', 'keypoint_name']).reset_index(drop=True)
        
        df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        
        print("=" * 80)
        print(f"✅ SUCESSO: Arquivo '{OUTPUT_EXCEL_PATH}' salvo com {len(df):,} registros.")
        print(f"   Frames processados: {frames_processados}")
        print(f"   Frames ignorados: {frames_ignorados}")
        
        print("\n📊 RESUMO POR CÂMERA (frames únicos detectados):")
        print("-" * 80)
        resumo = df.groupby('camera_id')['frame_id'].nunique().sort_index()
        for cam, num_frames in resumo.items():
            barra = "█" * int(num_frames / FRAMES_PER_CAMERA * 50)
            percentual = (num_frames / FRAMES_PER_CAMERA) * 100
            print(f"  {cam}: {num_frames:3d}/{FRAMES_PER_CAMERA} frames ({percentual:5.1f}%) {barra}")
        
        print("\n📈 RESUMO DE CONFIANÇA POR CÂMERA:")
        print("-" * 80)
        conf_resumo = df.groupby('camera_id')['confidence'].agg(['mean', 'min', 'max']).round(3)
        conf_resumo.columns = ['Média', 'Mínima', 'Máxima']
        print(conf_resumo.to_string())
        
        print("\n🎯 DISTRIBUIÇÃO DE KEYPOINTS:")
        print("-" * 80)
        kp_count = df.groupby('keypoint_name').size().sort_values(ascending=False)
        print(f"  Total de keypoints únicos detectados: {len(kp_count)}/{len(KEYPOINT_NAMES)}")
        print("\n  Top 10 keypoints mais detectados:")
        for kp, count in kp_count.head(10).items():
            print(f"    • {kp}: {count:,} detecções")
        
    except Exception as e:
        print(f"\n❌ ERRO ao salvar o arquivo Excel: {e}")
        return None

    return df

# Executa a função principal
if __name__ == "__main__":
    final_dataframe = processar_dados_alphapose()
    
    if final_dataframe is not None and not final_dataframe.empty:
        print("\n" + "=" * 80)
        print("🔍 VERIFICAÇÃO DE OFFSET (PERSPECTIVAS MÚLTIPLAS)")
        print("=" * 80)
        print("\nAmostra dos dados processados (primeiras 30 linhas):")
        print(final_dataframe.head(30).to_string(index=False))
        
        print("\n" + "-" * 80)
        print("🎥 TESTE DE SINCRONIZAÇÃO:")
        print("Frame_id '000' deve aparecer em todas as 16 câmeras (mesmo momento, 16 perspectivas)")
        print("-" * 80)
        frame_zero = final_dataframe[final_dataframe['frame_id'] == '000']['camera_id'].unique()
        print(f"Câmeras que detectaram frame_id '000': {sorted(frame_zero)}")
        print(f"Total: {len(frame_zero)}/{NUM_CAMERAS} câmeras")
        
        if len(frame_zero) == NUM_CAMERAS:
            print("✅ Sincronização perfeita!")
        else:
            print(f"⚠️ Atenção: {NUM_CAMERAS - len(frame_zero)} câmeras sem detecção no frame 000")
        
        print("\n" + "=" * 80)
        print("PROCESSAMENTO CONCLUÍDO!")
        print("=" * 80)