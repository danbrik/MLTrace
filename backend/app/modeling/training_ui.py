"""Training form metadata and active-field rules shared with the browser."""
from copy import deepcopy

GROUPS = [
    {'id': 'training', 'label': 'Trainingsablauf'}, {'id': 'optimizer', 'label': 'Optimizer'},
    {'id': 'loss', 'label': 'Loss'}, {'id': 'early', 'label': 'Early Stopping'},
    {'id': 'compute', 'label': 'Berechnung und Datenladen'},
]


def supports_validation(definition):
    return definition.training_mode == 'gradient' and definition.builder_kind != 'fast_anogan'


def field(path, values):
    return {'field': path, 'in': values}


def condition_matches(condition, parameters, method):
    if not condition:
        return True
    if 'all' in condition:
        return all(condition_matches(c, parameters, method) for c in condition['all'])
    if 'any' in condition:
        return any(condition_matches(c, parameters, method) for c in condition['any'])
    source, key = condition['field'].split('.', 1)
    value = (method if source == 'method' else parameters).get(key)
    if 'in' in condition:
        return value in condition['in']
    return isinstance(value, (int, float)) and value > condition['gt']


def training_schema(definition):
    schema = deepcopy(definition.training_schema)
    schema['ui_groups'] = GROUPS
    schema['supports_validation'] = supports_validation(definition)
    properties = schema.setdefault('properties', {})
    if supports_validation(definition):
        properties['model_selection'] = {'type': 'string', 'enum': ['last', 'best_validation'], 'label': 'Modell nach dem Training verwenden'}
    prediction = {'all': [field('method.prediction_branch', [True]), field('training.training_objective', ['reconstruction_prediction'])]}
    loss_conditions = [field('training.' + key, ['ssim', 'mae_ssim', 'mse_ssim'])
                       for key in ('loss', 'reconstruction_loss') if key in properties]
    combined_conditions = [field('training.' + key, ['mae_ssim', 'mse_ssim'])
                           for key in ('loss', 'reconstruction_loss') if key in properties]
    if 'prediction_loss' in properties:
        loss_conditions.append({'all': [prediction, field('training.prediction_loss', ['ssim', 'mae_ssim', 'mse_ssim'])]})
        combined_conditions.append({'all': [prediction, field('training.prediction_loss', ['mae_ssim', 'mse_ssim'])]})
    for key, prop in properties.items():
        group = 'training'
        if 'optimizer' in key or 'learning_rate' in key or key == 'weight_decay':
            group = 'optimizer'
        elif key.startswith('early_stopping'):
            group = 'early'
        elif key in {'num_workers', 'prefetch_factor', 'amp_enabled', 'log_interval_batches', 'log_interval_iterations'}:
            group = 'compute'
        elif 'loss' in key or key.startswith(('ssim_', 'kl_', 'beta', 'prediction_')) or key in {'gradient_penalty_lambda', 'kappa', 'encoder_training_mode', 'training_objective'}:
            group = 'loss'
        prop['ui_group'] = group
        if key == 'model_selection':
            prop['ui_group'] = 'validation'
        if key == 'validation_fraction':
            prop['visible_if'] = {'any': []}
        elif key.startswith('ssim_'):
            prop['visible_if'] = {'any': combined_conditions if key == 'ssim_weight' else loss_conditions}
        elif key.startswith('prediction_'):
            prop['visible_if'] = prediction if key != 'prediction_min_weight' else {'all': [prediction, field('training.prediction_weight_schedule', ['linear_decay', 'exponential_decay'])]}
        elif key == 'training_objective':
            prop['visible_if'] = field('method.prediction_branch', [True])
        elif key == 'early_stopping_patience':
            prop['visible_if'] = field('training.early_stopping_enabled', [True])
        elif key == 'prefetch_factor':
            prop['visible_if'] = {'field': 'training.num_workers', 'gt': 0}
        elif key == 'kappa':
            prop['visible_if'] = field('training.encoder_training_mode', ['izif'])
        if key == 'amp_enabled':
            prop['description'] = 'Automatische gemischte Genauigkeit; wirkt ausschließlich bei CUDA.'
    return schema


def active_parameters(definition, parameters, method):
    from app.modeling.base import validate_schema_values
    schema = training_schema(definition)
    values = {**{key: prop['default'] for key, prop in schema.get('properties', {}).items() if 'default' in prop}, **definition.default_training_config, **parameters}
    active = {key: prop for key, prop in schema.get('properties', {}).items()
              if condition_matches(prop.get('visible_if'), values, method)}
    result = {key: value for key, value in values.items() if key in active}
    import math
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{key}: Bitte eine endliche Zahl angeben.')
    validate_schema_values({'properties': active, 'required': [key for key in schema.get('required', []) if key in active]}, result, 'training_parameters')
    # Omitted branches must not accidentally activate runtime fallbacks.
    if 'training_objective' in schema.get('properties', {}) and not method.get('prediction_branch'):
        result['training_objective'] = 'reconstruction'
    return result
