# Ferramenta de Análise de Composição de Câmeras (AlphaPose)

Este projeto contém um conjunto de scripts para processar a saída JSON do AlphaPose e uma aplicação web interativa (Streamlit) para analisar a melhor composição de câmeras para detecção de keypoints.

## 🚀 Guia Rápido

Siga estes passos para configurar e executar a análise.

### 1. Configurar o Ambiente

Você precisa criar um ambiente virtual isolado para este projeto.

**Usando `venv` (Recomendado):**
```bash
# Crie o ambiente
python3 -m venv venv

# Ative o ambiente
source venv/bin/activate
````

**Usando `conda`:**

```bash
# Crie o ambiente (substitua 'meu_env' pelo nome que desejar)
conda create -n meu_env python=3.10

# Ative o ambiente
conda activate meu_env
```

### 2\. Instalar as Dependências

Com o ambiente ativado, instale todas as bibliotecas necessárias de uma só vez:

```bash
pip install -r requirements.txt
```

### 3\. Processar os Dados Brutos

Antes de rodar a aplicação, você deve converter seu `alphapose-results.json` em um arquivo Excel formatado.

**Importante:** Coloque seu arquivo `alphapose-results.json` na mesma pasta do script.

```bash
python json_processor.py
```

*(Este script irá gerar o arquivo `dados_processados_por_camera.xlsx`)*

### 4\. Executar o Dashboard de Análise

Agora, inicie a aplicação web interativa com o Streamlit:

```bash
streamlit run analise_dashboard.py
```

O Streamlit abrirá automaticamente uma aba no seu navegador com o dashboard pronto para uso.

```
