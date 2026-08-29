# 🚀 Instruccions d'Entrenament a Google Colab

## ⚡ MILLORES IMPLEMENTADES

Hem optimitzat el codi d'entrenament per **reduir dràsticament el temps per època**:

1. ✅ **Dataset copiat a `/content/` local** - Evita latència de Google Drive
2. ✅ **Sliding Window amb overlap=0.25** - Redueix patches generats per validació
3. ✅ **Validació cada 5 èpoques** - Estalvia temps sense perdre informació
4. ✅ **Batch size 1 per validació** - Evita OOM amb Sliding Window
5. ✅ **num_workers=4 + pin_memory=True** - Millor paral·lelització GPU

**Resultat esperat:** Temps per època reduït de ~40min a ~8-12min

---

## 📋 CEL·LES DE COLAB

Copia i enganxa aquestes cel·les al teu Google Colab **en aquest ordre**:

### CEL·LA 1: Muntar Drive i Copiar Dataset

```python
from google.colab import drive
drive.mount("/content/drive")

# Copiar dataset a disc local del Colab per velocitat
print("Copiant dataset a disc local...")
!cp -r /content/drive/MyDrive/BrainTumorProject/dataset/TrainingData /content/TrainingData
print("Dataset copiat!")
```

---

### CEL·LA 2: Clonar Repositori i Instal·lar Dependències

```python
!git clone https://github.com/bielvicens/Brain-Tumor-3D-Segmentation.git
%cd Brain-Tumor-3D-Segmentation
!pip install -r requirements.txt
```

---

### CEL·LA 3: Verificar Setup

```python
import torch
from pathlib import Path

print("=" * 60)
print("VERIFICACIÓ DE L'ENTORN")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print("GPU:", gpu.name)
    print("VRAM:", f"{gpu.total_memory / 1024**3:.2f} GB")
else:
    print("Running on CPU")

# Verificar dataset
dataset = Path("/content/TrainingData")
print("\nDataset exists:", dataset.exists())

if dataset.exists():
    patients = sorted(dataset.iterdir())
    print(f"Pacients disponibles: {len(patients)}")
    print(f"Primers 5: {[p.name for p in patients[:5]]}")

print("=" * 60)
```

---

### CEL·LA 4: Configuració

```python
from src.utils import ProjectConfig

config = ProjectConfig()

# IMPORTANT: Apuntar al dataset local copiat
config.data.dataset_root = "/content/TrainingData"

# IMPORTANT: Checkpoints al Drive per persistència
config.checkpoint.directory = (
    "/content/drive/MyDrive/BrainTumorProject/checkpoints"
)

# Optimitzacions de velocitat
config.data.num_workers = 4
config.data.pin_memory = True

print("=" * 60)
print("CONFIGURACIÓ")
print("=" * 60)
print("Dataset root:", config.data.dataset_root)
print("Checkpoint dir:", config.checkpoint.directory)
print("Num workers:", config.data.num_workers)
print("Pin memory:", config.data.pin_memory)
print("Batch size:", config.training.batch_size)
print("Learning rate:", config.training.learning_rate)
print("Epochs:", config.training.epochs)
print("=" * 60)
```

---

### CEL·LA 5: Build Datasets

```python
from src.builders import (
    build_pipeline,
    build_datasets,
    build_model,
)

pipeline = build_pipeline(
    config,
    training=True,
)

train_dataset, validation_dataset = build_datasets(
    config,
    train_pipeline=pipeline,
    validation_pipeline=build_pipeline(
        config,
        training=False,
    ),
)

print("\n" + "=" * 60)
print("DATASETS")
print("=" * 60)
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(validation_dataset))
print("=" * 60)
```

**⚠️ OPCIONAL - DEBUGGING MODE:**
```python
# DESCOMENTA NOMÉS SI VOLS FER UN TEST RÀPID
# ATENCIÓ: Això fa que les mètriques no siguin representatives!

# train_dataset.patient_ids = train_dataset.patient_ids[:10]
# validation_dataset.patient_ids = validation_dataset.patient_ids[:2]
# print("\n⚠️ ATENCIÓ: Dataset limitat per debugging!")
```

---

### CEL·LA 6: Build DataLoaders

```python
from src.builders import (
    build_dataloader,
    build_loss,
    build_optimizer,
    build_scheduler,
)

train_loader = build_dataloader(
    train_dataset,
    config,
)

# IMPORTANT: Batch size 1 per validació per evitar OOM amb Sliding Window
validation_loader = build_dataloader(
    validation_dataset,
    config,
    batch_size=1,
)

print("\n" + "=" * 60)
print("DATALOADERS")
print("=" * 60)
print("Train batches:", len(train_loader))
print("Train batch size:", train_loader.batch_size)
print("Validation batches:", len(validation_loader))
print("Validation batch size:", validation_loader.batch_size)
print("=" * 60)
```

---

### CEL·LA 7: Build Model i Components

```python
model = build_model(config)

optimizer = build_optimizer(
    model,
    config,
)

criterion = build_loss(config)
criterion = criterion.to(config.training.device)

scheduler = build_scheduler(optimizer)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)
trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)
print("Model:", type(model).__name__)
print("Total parameters:", f"{total_parameters:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")
print("=" * 60)

print("\n" + "=" * 60)
print("OPTIMIZER + LOSS")
print("=" * 60)
print("Optimizer:", type(optimizer).__name__)
print("Loss:", type(criterion).__name__)
print("Scheduler:", type(scheduler).__name__)
print("=" * 60)
```

---

### CEL·LA 8: Build Trainer

```python
from src.models import Trainer, TrainingHistory
from src.utils import EarlyStopping

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=config.training.device,
)

early_stopping = EarlyStopping(
    patience=config.early_stopping.patience,
    min_delta=config.early_stopping.min_delta,
    mode="max",
)

print("\n" + "=" * 60)
print("TRAINER")
print("=" * 60)
print("Device:", trainer.device)
print("AMP enabled:", trainer.scaler.is_enabled())
if trainer.device.type == "cuda":
    print("Mixed precision: FP16")
print("=" * 60)
```

---

### CEL·LA 9: Checkpoint Setup i Resume

```python
from pathlib import Path

checkpoint_dir = (
    Path(config.checkpoint.directory)
    / config.experiment.name
    / "randomcrop3d_slidingwindow_optimized"
)
checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)

last_checkpoint = checkpoint_dir / "last.pt"

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)
print("Checkpoint directory:")
print(checkpoint_dir)
print("Resume checkpoint:")
print(last_checkpoint)
print("=" * 60)

# Resume si existeix checkpoint
if last_checkpoint.exists():
    print("\n📂 Carregant checkpoint...")
    checkpoint = torch.load(
        last_checkpoint,
        map_location=trainer.device,
    )

    trainer.model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    trainer.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = int(checkpoint["epoch"])
    
    checkpoint_history = checkpoint.get("history", None)
    if checkpoint_history is None:
        raise ValueError(
            "Checkpoint does not contain training history."
        )

    history = TrainingHistory(
        train_loss=checkpoint_history["train_loss"],
        val_loss=checkpoint_history["val_loss"],
        train_dice=checkpoint_history["train_dice"],
        val_dice=checkpoint_history["val_dice"],
        train_ncr_dice=checkpoint_history["train_ncr_dice"],
        val_ncr_dice=checkpoint_history["val_ncr_dice"],
        train_ed_dice=checkpoint_history["train_ed_dice"],
        val_ed_dice=checkpoint_history["val_ed_dice"],
        train_et_dice=checkpoint_history["train_et_dice"],
        val_et_dice=checkpoint_history["val_et_dice"],
    )

    print("✅ Checkpoint carregat correctament!")
    print(f"Última època completada: {start_epoch}")
    print(f"Reprenent des de l'època: {start_epoch + 1}")
    
    if history.val_ncr_dice:
        valid_ncr = [x for x in history.val_ncr_dice if x is not None]
        if valid_ncr:
            print(f"Millor NCR Dice fins ara: {max(valid_ncr):.4f}")
    
    # Restaurar estat d'early stopping
    if history.val_ncr_dice:
        for ncr_dice in history.val_ncr_dice:
            if ncr_dice is not None:
                early_stopping.step(ncr_dice)

else:
    print("\n📝 No s'ha trobat last.pt")
    print("Començant entrenament des de zero...")
    
    start_epoch = 0
    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_dice=[],
        val_dice=[],
        train_ncr_dice=[],
        val_ncr_dice=[],
        train_ed_dice=[],
        val_ed_dice=[],
        train_et_dice=[],
        val_et_dice=[],
    )
```

---

### CEL·LA 10: 🔥 ENTRENAMENT PRINCIPAL

```python
print("\n" + "=" * 60)
print("COMENÇANT ENTRENAMENT")
print("=" * 60)
print(f"Època inicial: {start_epoch + 1}")
print(f"Èpoques totals: {config.training.epochs}")
print(f"Èpoques restants: {config.training.epochs - start_epoch}")
print("\n🔧 OPTIMITZACIONS ACTIVES:")
print("  - Sliding Window overlap: 0.25 (velocitat)")
print("  - Validació cada 5 èpoques")
print("  - Batch size validació: 1 (estabilitat)")
print("  - Workers: 4 (paral·lelització)")
print("=" * 60)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=validation_loader,
    epochs=config.training.epochs,
    start_epoch=start_epoch,
    history=history,
    early_stopping=early_stopping,
    checkpoint_dir=checkpoint_dir,
    scheduler=scheduler,
    val_every_n_epochs=5,
    sliding_window_overlap=0.25,
)
```

---

### CEL·LA 11: Resultats Finals

```python
print("\n" + "=" * 60)
print("🎉 ENTRENAMENT FINALITZAT")
print("=" * 60)
print(f"Èpoques completades: {history.epochs}")

# Filtrar valors None abans de calcular màxims
valid_val_ncr = [x for x in history.val_ncr_dice if x is not None]
valid_val_dice = [x for x in history.val_dice if x is not None]

if valid_val_ncr:
    best_ncr_dice = max(valid_val_ncr)
    best_ncr_epoch = history.val_ncr_dice.index(best_ncr_dice) + 1
    print(f"\n🏆 Millor validació NCR Dice: {best_ncr_dice:.4f}")
    print(f"   (època {best_ncr_epoch})")

if valid_val_dice:
    best_mean_dice = max(valid_val_dice)
    best_mean_epoch = history.val_dice.index(best_mean_dice) + 1
    print(f"\n🏆 Millor validació Mean Dice: {best_mean_dice:.4f}")
    print(f"   (època {best_mean_epoch})")

if history.train_loss:
    print(f"\nLoss final train: {history.train_loss[-1]:.6f}")

if history.val_loss:
    last_val_loss = next(
        (x for x in reversed(history.val_loss) if x is not None),
        None
    )
    if last_val_loss is not None:
        print(f"Loss final val: {last_val_loss:.6f}")

print("\n📁 Checkpoints guardats a:")
print(f"   {checkpoint_dir}")
print(f"   - best.pt (millor model segons NCR Dice)")
print(f"   - last.pt (últim estat, per resumir)")
print("=" * 60)
```

---

## 🎯 PARÀMETRES CONFIGURABLES

Si vols ajustar les optimitzacions, modifica aquests valors a la **CEL·LA 10**:

```python
history = trainer.fit(
    ...
    val_every_n_epochs=5,        # Canvia a 1 per validar cada època
    sliding_window_overlap=0.25,  # Canvia a 0.5 per més precisió (més lent)
)
```

**Recomanacions:**
- `val_every_n_epochs=5` + `overlap=0.25` → Màxima velocitat (~8-12 min/època)
- `val_every_n_epochs=1` + `overlap=0.5` → Màxima precisió (~35-40 min/època)
- **Compromís recomanat:** Valors actuals (5 i 0.25)

---

## 📊 RESULTATS ESPERATS

Amb aquestes optimitzacions i el dataset complet (1000 train + 251 val):

- **Temps per època (train):** ~3-5 min
- **Temps per època (train + val):** ~8-12 min cada 5 èpoques
- **Validació estable:** Sense salts bruscos (0.3 → 0.6)
- **NCR Dice esperat:** 0.65-0.70 (millor que l'anterior 0.66)

---

## 🐛 TROUBLESHOOTING

**Si surten errors de memòria (OOM):**
- Redueix `config.training.batch_size` a 1
- Assegura't que `validation_loader` té `batch_size=1`

**Si l'entrenament és massa lent:**
- Augmenta `val_every_n_epochs` a 10
- Redueix `sliding_window_overlap` a 0.20

**Si la validació dona errors:**
- Comprova que el Sliding Window està implementat correctament
- Verifica que el dataset no està buit

---

## ✅ CHECKLIST ABANS D'ENTRENAR

- [ ] Dataset copiat a `/content/TrainingData`
- [ ] Checkpoints configurats al Drive
- [ ] `num_workers=4` i `pin_memory=True`
- [ ] Validation loader amb `batch_size=1`
- [ ] `val_every_n_epochs=5` configurat
- [ ] `sliding_window_overlap=0.25` configurat
- [ ] Comentades les línies `[:10]` i `[:2]` (si volies dataset complet)

---

**Data última actualització:** 29 Agost 2026
**Autor:** Kiro AI + Lluis Vicens
