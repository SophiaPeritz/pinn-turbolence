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