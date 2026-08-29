"""
Codi d'entrenament optimitzat per Google Colab.

MILLORES IMPLEMENTADES:
1. Dataset copiat a /content/ local per evitar latència de Google Drive
2. Sliding Window amb overlap=0.25 per accelerar validació
3. Validació cada 5 èpoques (configurable)
4. Batch size 1 per validació per evitar OOM
5. num_workers=4 i pin_memory=True per paral·lelització

INSTRUCCIONS:
- Copia aquest fitxer al teu Colab
- Executa cel·la per cel·la
- Els checkpoints es guarden al Drive per persistència
"""

# ============================================================
# CEL·LA 1: MUNTAR DRIVE I COPIAR DATASET
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

# Copiar dataset a disc local del Colab per velocitat
print("Copiant dataset a disc local...")
!cp -r /content/drive/MyDrive/BrainTumorProject/dataset/TrainingData /content/TrainingData
print("Dataset copiat!")


# ============================================================
# CEL·LA 2: CLONAR REPOSITORI I INSTAL·LAR DEPENDÈNCIES
# ============================================================

!git clone https://github.com/bielvicens/Brain-Tumor-3D-Segmentation.git
%cd Brain-Tumor-3D-Segmentation
!pip install -r requirements.txt


# ============================================================
# CEL·LA 3: VERIFICAR SETUP
# ============================================================

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


# ============================================================
# CEL·LA 4: CONFIGURACIÓ
# ============================================================

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


# ============================================================
# CEL·LA 5: BUILD DATASETS
# ============================================================

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

# ============================================================
# OPCIONAL: LIMITAR DATASET PER DEBUGGING
# ============================================================
# DESCOMENTA NOMÉS SI VOLS FER UN TEST RÀPID
# ATENCIÓ: Això fa que les mètriques no siguin representatives!
#
# train_dataset.patient_ids = train_dataset.patient_ids[:10]
# validation_dataset.patient_ids = validation_dataset.patient_ids[:2]
# print("\n⚠️ ATENCIÓ: Dataset limitat per debugging!")
# print(f"Train: {len(train_dataset)} pacients")
# print(f"Val: {len(validation_dataset)} pacients")


# ============================================================
# CEL·LA 6: BUILD DATALOADERS
# ============================================================

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


# ============================================================
# CEL·LA 7: BUILD MODEL I COMPONENTS
# ============================================================

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


# ============================================================
# CEL·LA 8: BUILD TRAINER
# ============================================================

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


# ============================================================
# CEL·LA 9: CHECKPOINT SETUP I RESUME
# ============================================================

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
        # Filtrar valors None
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


# ============================================================
# CEL·LA 10: ENTRENAMENT PRINCIPAL
# ============================================================

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
    val_every_n_epochs=5,  # Validar cada 5 èpoques
    sliding_window_overlap=0.25,  # Overlap baix per velocitat
)


# ============================================================
# CEL·LA 11: RESULTATS FINALS
# ============================================================

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
    # Trobar l'última loss de validació no None
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
