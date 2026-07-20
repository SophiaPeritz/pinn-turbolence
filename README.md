# pinn-turbolence

Progetto: PINN per il flusso di Kolmogorov (Navier–Stokes 2D incomprimibile).

Contenuto principale:
- `src/` : implementazioni (rete, loss, training, utils, PDE helper)
- `configs/kolmogorov.yaml` : configurazione d'esempio
- `notebooks/train_colab.ipynb` : notebook con esempio d'uso
- `results/` : output e checkpoint (peso baseline incluso)

Quick start

1) Installare dipendenze (consigliato in virtualenv):

```bash
pip install -r requirements.txt
# opzionale: pip install tensorboard
```

2) Eseguire il training con il config di esempio:

```bash
python scripts/train.py --config configs/kolmogorov.yaml
```

3) Valutare uno checkpoint salvato:

```bash
python scripts/evaluate.py --checkpoint results/weights/baseline_final.pt \
	--config configs/kolmogorov.yaml
```

### Strato 2: causal training

La configurazione causale divide i punti PDE, ordinati nel tempo, in chunk
disgiunti. Ogni chunk riceve un peso esponenziale determinato dai residui dei
chunk precedenti, come in *Simulating Three-dimensional Turbulence with
Physics-informed Neural Networks* (Wang et al., 2025).

```bash
python scripts/train.py --config configs/kolmogorov_causal.yaml
```

Gli output sono salvati in `results/causal`. Su Colab, dopo aver caricato il
file YAML, impostare la directory su Google Drive prima di chiamare `train`:

```python
cfg["results_dir"] = DRIVE_DIR + "/causal"
```

TensorBoard registra anche `causal/min_weight` e `causal/mean_weight`, utili
per osservare il progressivo sblocco dei chunk temporali più avanzati.

Note rapide
- Il notebook `notebooks/train_colab.ipynb` mostra un esempio passo-passo.
- I checkpoint salvano anche una copia del `configs/kolmogorov.yaml` usato.
- Per il logging usare TensorBoard (se installato):

```bash
tensorboard --logdir results/tensorboard
```

Contribuire
- Aggiungere issue/PR, includere un file `configs/` per i nuovi esperimenti.

Licenza: MIT (file LICENSE incluso).
