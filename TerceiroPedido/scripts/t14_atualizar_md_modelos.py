#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
MD = ROOT / "TerceiroPedido.md"
MET = ROOT / "dados" / "metricas_dl_lstm_gru.csv"

def main():
    if not MD.exists():
        raise FileNotFoundError(f"Não encontrei: {MD} (rode o explorar_md.py antes)")

    if not MET.exists():
        raise FileNotFoundError(f"Não encontrei: {MET} (rode o T12 antes)")

    met = pd.read_csv(MET)
    met_md = met.to_markdown(index=False)

    bloco = []
    bloco.append("\n## Modelos de previsão (Deep Learning) — LSTM e GRU (PyTorch)\n")
    bloco.append("Nesta etapa foram treinados modelos **LSTM** e **GRU** usando divisão temporal **sem vazamento**: "
                 "para prever o dia *t*, o modelo recebe apenas informações históricas até *t-1* (janela lookback) "
                 "e variáveis meteorológicas/vento como covariáveis.\n")
    bloco.append("### Métricas (treino e teste)\n")
    bloco.append(met_md + "\n")
    bloco.append("### Artefatos\n")
    bloco.append("- `dados/previsoes_dl_lstm_gru.csv`\n"
                 "- `dados/metricas_dl_lstm_gru.csv`\n"
                 "- `graficos/T12_modelos_dl/previsao_dl_zoom_1ano.png`\n"
                 "- `graficos/T13_comparacao/comparacao_previsoes_zoom_1ano.png`\n")
    bloco.append("\n**Resumo:** no conjunto de teste, os modelos LSTM/GRU superaram as baselines (persistência e MM7), "
                 "com destaque para a **GRU**.\n")

    txt = MD.read_text(encoding="utf-8")
    if "## Modelos de previsão (Deep Learning)" in txt:
        # Se já existir, não duplica
        print("[OK] Seção de modelos já existe no markdown. Nada a fazer.")
        return

    MD.write_text(txt + "\n" + "\n".join(bloco), encoding="utf-8")
    print("[OK] Markdown atualizado:", MD)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
