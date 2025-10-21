import json
import pandas as pd
import os

# --------------------------------------------------------------------------
# --- CONFIGURAÇÕES FINAIS ---
# --------------------------------------------------------------------------
VIDEO_SECONDS = 40
FPS = 30
NUM_CAMERAS = 6
INPUT_JSON_PATH = 'alphapose-results.json'
OUTPUT_EXCEL_PATH = 'dados_processados_por_camera.xlsx'
FRAME_PADDING = 4  # Para formatar 0000, 0001, etc.

# --------------------------------------------------------------------------
# --- LÓGICA PRINCIPAL (CORRIGIDA COM OFFSET) ---
# --------------------------------------------------------------------------

def processar_dados_alphapose():
    """
    Lê o JSON, calcula o ID da câmera (offset) e o FRAME DE REFERÊNCIA (módulo)
    para comparar corretamente as perspectivas e salva em Excel.
    
    IMPORTANTE: O vídeo tem 7200 frames totais (40s * 30fps * 6 câmeras)
    - Frames 0-1199: cam_01 (mesma cena, perspectiva 1)
    - Frames 1200-2399: cam_02 (mesma cena, perspectiva 2)
    - Frames 2400-3599: cam_03 (mesma cena, perspectiva 3)
    - Frames 3600-4799: cam_04 (mesma cena, perspectiva 4)
    - Frames 4800-5999: cam_05 (mesma cena, perspectiva 5)
    - Frames 6000-7199: cam_06 (mesma cena, perspectiva 6)
    
    O frame_id normalizado (0-1199) representa o MESMO MOMENTO no tempo.
    """
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    if not os.path.exists(INPUT_JSON_PATH):
        print(f"ERRO: Arquivo de entrada '{INPUT_JSON_PATH}' não encontrado.")
        return

    frames_per_camera = VIDEO_SECONDS * FPS
    print(f"Configuração: {NUM_CAMERAS} câmeras, {VIDEO_SECONDS}s por vídeo, {FPS} FPS.")
    print(f"Calculado: {frames_per_camera} frames por câmera (1200 frames = 1 perspectiva completa).")
    print(f"Total esperado: {frames_per_camera * NUM_CAMERAS} frames (7200 frames).")
    print("-" * 80)

    try:
        print(f"Carregando JSON de '{INPUT_JSON_PATH}'...")
        with open(INPUT_JSON_PATH, 'r') as f:
            detections = json.load(f)
        print(f"Total de {len(detections)} detecções encontradas no JSON.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo JSON: {e}")
        return

    processed_data = []
    print("Processando dados com offset de câmeras...")

    for detection in detections:
        original_frame_id = detection.get('image_id', '0.jpg')
        
        try:
            base_name, extension = os.path.splitext(original_frame_id)
            frame_number = int(base_name)

            # --- CORREÇÃO LÓGICA 1: CALCULAR O ÍNDICE DA CÂMERA ---
            # Cada bloco de 1200 frames = 1 câmera
            camera_index = frame_number // frames_per_camera
            
            # --- CORREÇÃO LÓGICA 2: FILTRAR FRAMES INVÁLIDOS ---
            if camera_index >= NUM_CAMERAS:
                print(f"Aviso: Ignorando frame '{original_frame_id}' (câmera inválida: {camera_index + 1}).")
                continue
                
            camera_id = f"cam_{camera_index + 1:02d}"

            # --- CORREÇÃO LÓGICA 3: CALCULAR O FRAME DE REFERÊNCIA ---
            # O momento no tempo (0-1199) é o mesmo para todas as câmeras
            ref_frame_num = frame_number % frames_per_camera
            ref_frame_id = f"{ref_frame_num:0{FRAME_PADDING}d}"

            keypoints = detection.get('keypoints', [])
            score = detection.get('score', 0.0)
            
            for kp_idx in range(0, len(keypoints), 3):
                point_index = kp_idx // 3
                if point_index < len(keypoint_names):
                    keypoint_name = keypoint_names[point_index]
                    x = keypoints[kp_idx]
                    y = keypoints[kp_idx + 1]
                    confidence = keypoints[kp_idx + 2]
                    
                    processed_data.append({
                        "camera_id": camera_id,
                        "frame_id": ref_frame_id,  # Frame normalizado (0-1199)
                        "original_frame_id": original_frame_id,  # Frame original (0-7199)
                        "keypoint_name": keypoint_name,
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                        "detection_score": score  # Score geral da detecção
                    })
        except (ValueError, TypeError) as e:
            print(f"Aviso: Ignorando detecção com image_id inválido: {original_frame_id} ({e})")
            continue

    df = pd.DataFrame(processed_data)

    if df.empty:
        print("ERRO: Nenhum dado foi processado. Verifique o formato do JSON.")
        return None

    try:
        print(f"\nSalvando dados em formato Excel em '{OUTPUT_EXCEL_PATH}'...")
        
        # Reordenando colunas para maior clareza
        colunas_ordenadas = [
            "camera_id", "frame_id", "keypoint_name", 
            "confidence", "x", "y", "detection_score", "original_frame_id"
        ]
        df = df[colunas_ordenadas]
        
        # Ordena por câmera e frame para facilitar análise
        df = df.sort_values(['camera_id', 'frame_id', 'keypoint_name']).reset_index(drop=True)
        
        df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        
        print("-" * 80)
        print(f"SUCESSO: Arquivo '{OUTPUT_EXCEL_PATH}' salvo com {len(df)} registros.")
        print("\n--- Resumo por Câmera (frames únicos detectados) ---")
        resumo = df.groupby('camera_id')['frame_id'].nunique().sort_index()
        for cam, num_frames in resumo.items():
            print(f"{cam}: {num_frames} frames únicos (esperado: 1200)")
        
        print("\n--- Resumo de Confiança por Câmera ---")
        conf_resumo = df.groupby('camera_id')['confidence'].agg(['mean', 'min', 'max']).round(3)
        print(conf_resumo)
        
    except Exception as e:
        print(f"ERRO ao salvar o arquivo Excel: {e}")
        return None

    return df

# Executa a função principal
if __name__ == "__main__":
    final_dataframe = processar_dados_alphapose()
    if final_dataframe is not None and not final_dataframe.empty:
        print("\n--- Amostra dos Dados Processados ---")
        print(final_dataframe.head(20))
        print("\n--- Verificação de Offset ---")
        print("Exemplo: frame_id '0000' deve aparecer em todas as 6 câmeras (mesmo momento, 6 perspectivas)")
        frame_zero = final_dataframe[final_dataframe['frame_id'] == '0000']['camera_id'].unique()
        print(f"Câmeras que detectaram frame_id '0000': {sorted(frame_zero)}")