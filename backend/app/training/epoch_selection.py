"""Choose an inference artifact independently of the resumable training state."""
import math


class EpochSelection:
    def __init__(self, parameters, has_validation):
        self.mode = parameters.get('model_selection', 'last')
        if self.mode not in {'last', 'best_validation'}:
            raise ValueError('Unknown model epoch selection.')
        if self.mode == 'best_validation' and not has_validation:
            raise ValueError('Beste Validierungsepoche erfordert eine aktive Validierung.')
        self.best_loss = None
        self.best_epoch = None
        self.best_state = None
        self.last_epoch = None

    def restore(self, checkpoint):
        self.last_epoch = int(checkpoint['epoch'])
        self.best_loss = checkpoint.get('best_val_loss')
        self.best_epoch = checkpoint.get('best_epoch')
        self.best_state = checkpoint.get('best_model_state_dict')
        if self.mode == 'best_validation' and (self.best_state is None or self.best_epoch is None):
            raise ValueError('Dieser Checkpoint enthält keine besten Modellgewichte. Bitte einen neuen Lauf starten.')

    def observe(self, epoch, loss, model):
        if loss is not None and not math.isfinite(loss):
            raise ValueError('Der Validierungsverlust ist nicht endlich. Es wird kein fertiges Modell freigegeben.')
        self.last_epoch = epoch
        if loss is not None and (self.best_loss is None or loss < self.best_loss):
            self.best_loss, self.best_epoch = loss, epoch
            if self.mode == 'best_validation':
                # Clone tensors, including buffers: state_dict itself is only a live view.
                from copy import deepcopy
                from collections import OrderedDict
                state = model.state_dict()
                self.best_state = OrderedDict((key, value.detach().cpu().clone() if hasattr(value, 'detach') else deepcopy(value)) for key, value in state.items())
                if hasattr(state, '_metadata'):
                    self.best_state._metadata = deepcopy(state._metadata)

    def checkpoint_fields(self):
        return {'best_epoch': self.best_epoch, 'best_model_state_dict': self.best_state}

    def finish(self, model):
        if self.mode == 'best_validation':
            if self.best_state is None:
                raise ValueError('Keine gültigen Validierungsgewichte verfügbar.')
            model.load_state_dict(self.best_state)
            return self.best_epoch
        return self.last_epoch
