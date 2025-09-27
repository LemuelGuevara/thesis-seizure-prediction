from src.logger import setup_logger

logger = setup_logger(name="early_stopping")


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()

        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f"No improvement. EarlyStopping counter: {self.counter} out of {self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs without improvement"
                )
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            if self.counter > 0:
                logger.info("Performance improved - resetting patience counter")
            self.counter = 0
