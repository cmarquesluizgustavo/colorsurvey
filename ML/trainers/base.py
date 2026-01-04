from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer interface for all training methods."""
    
    def __init__(self, config, data_bundle, logger):
        self.config = config
        self.data_bundle = data_bundle
        self.logger = logger
    
    @abstractmethod
    def setup(self):
        """Initialize models, optimizers, and loss functions."""
        pass
    
    @abstractmethod
    def train(self):
        """Execute the training loop."""
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs):
        """Evaluate the model. Returns dict of metrics."""
        pass
    
    @abstractmethod
    def save_model(self):
        """Save trained models."""
        pass


class TrainerFactory:
    """Factory to create trainers based on type."""
    
    _trainers = {}
    
    @classmethod
    def register(cls, name, trainer_class):
        """Register a trainer class."""
        cls._trainers[name] = trainer_class
    
    @classmethod
    def create(cls, trainer_type, config, data_bundle, logger):
        """Create and return a trainer instance."""
        trainer_class = cls._trainers.get(trainer_type)
        if not trainer_class:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        return trainer_class(config, data_bundle, logger)
