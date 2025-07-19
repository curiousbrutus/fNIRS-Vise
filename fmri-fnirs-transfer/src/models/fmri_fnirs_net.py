import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

class FmriGuidedFnirsNet(LightningModule):
    """fMRI-guided fNIRS transfer learning model."""
    def __init__(
        self, 
        fnirs_channels: int = 52,
        fmri_dim: int = 768,
        num_classes: int = 4,
        transfer_mode: str = "feature_guided",
        freeze_fmri: bool = True,
        distill_alpha: float = 0.5,
        fmri_encoder_path: str = "checkpoints/fmri_encoder.pth"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fnirs_channels = fnirs_channels
        self.fmri_dim = fmri_dim
        self.num_classes = num_classes
        self.transfer_mode = transfer_mode
        self.freeze_fmri = freeze_fmri
        self.distill_alpha = distill_alpha
        self.fmri_encoder_path = fmri_encoder_path

        # Load fMRI encoder and freeze if required
        self.fmri_encoder = nn.Linear(self.fmri_dim, 128) # Dummy fMRI encoder
        if os.path.exists(self.fmri_encoder_path):
            # In a real scenario, load the actual pre-trained weights
            print(f"Loading fMRI encoder from {self.fmri_encoder_path} (stub)")
            # self.fmri_encoder.load_state_dict(torch.load(self.fmri_encoder_path))
        
        if self.freeze_fmri:
            for param in self.fmri_encoder.parameters():
                param.requires_grad = False

        # fNIRS encoder: 3x Conv1D (kernel=7) -> GAP -> 128-D
        self.fnirs_encoder = nn.Sequential(
            nn.Conv1d(self.fnirs_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Classifier
        self.classifier = nn.Linear(128 + 128, self.num_classes)

        self.criterion = nn.CrossEntropyLoss() # Example loss function for classification
        self.distill_criterion = nn.MSELoss() # Example distillation loss
        self.scaler = GradScaler()

    def forward(self, fmri_data, fnirs_data):
        """Forward pass based on the selected mode."""
        fmri_features = self.fmri_encoder(fmri_data)
        fnirs_features = self.fnirs_encoder(fnirs_data)

        if self.transfer_mode == 'feature_guided':
            combined_input = torch.cat([fmri_features, fnirs_features], dim=1)
            output = self.classifier(combined_input)
        elif self.transfer_mode == 'distill':
            # Teacher (fMRI encoder) is used for distillation
            with torch.no_grad():
                teacher_output = self.fmri_encoder(fmri_data) # Assuming teacher output is fMRI features
            student_output = fnirs_features
            combined_input = torch.cat([fmri_features, student_output], dim=1)
            output = self.classifier(combined_input) # Classifier on combined features

        elif self.transfer_mode == 'finetune':
             combined_input = torch.cat([fmri_features, fnirs_features], dim=1)
             output = self.classifier(combined_input)

        else:
            raise ValueError(f"Unknown transfer mode: {self.transfer_mode}")

        return output

    def training_step(self, batch, batch_idx):
        fmri_data = batch['fmri']
        fnirs_data = batch['fnirs']
        labels = batch['label']

        with autocast():
            outputs = self(fmri_data, fnirs_data)
            loss = self.criterion(outputs, labels)

            if self.transfer_mode == 'distill':
                with torch.no_grad():
                    teacher_output = self.fmri_encoder(fmri_data)
                student_output = self.fnirs_encoder(fnirs_data)
                distill_loss = self.distill_criterion(student_output, teacher_output)
                loss = (1 - self.distill_alpha) * loss + self.distill_alpha * distill_loss

        self.scaler.scale(loss).backward()
        # self.scaler.step(self.trainer.optimizers[0]) # Optimizer step is handled by Lightning
        # self.scaler.update() # Scaler update is handled by Lightning

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        # Add CosineAnnealingLR if needed, requires trainer.max_epochs
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # return [optimizer], [scheduler]
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Gradient checkpointing (manual for K80 if needed)
        pass
