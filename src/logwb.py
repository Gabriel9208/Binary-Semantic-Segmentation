import wandb

class LogWB:
    def __init__(self, name, hyperparam):
        self.name = name
        self.run = wandb.init(
            entity="gabrieee-national-taiwan-university-of-science-and-techn",
            project="NYCU LAB2",
            name=name,
            config=hyperparam,
        )
        
    def write_checkpoint(self, file_path):
        artifact = wandb.Artifact(
            name=self.name, 
            type="model"
        )
        
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
        print(f"Checkpoint uploaded to WandB.")
        
    def log_data(self, e, train_loss, val_loss, current_lr):
        log_data = {
                    "epoch": e,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr, 
                }
    
        wandb.log(log_data)