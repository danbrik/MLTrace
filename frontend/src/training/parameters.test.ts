import { describe, expect, it } from 'vitest';
import { activeTrainingKeys, conditionMatches } from './parameters';
import type { ConfigSchema } from '../types';
const schema: ConfigSchema = { type: 'object', properties: {
  loss: { type: 'string' },
  ssim_window_size: { type: 'integer', visible_if: { field: 'training.loss', in: ['ssim', 'mse_ssim'] } },
  ssim_weight: { type: 'number', visible_if: { field: 'training.loss', in: ['mse_ssim'] } },
  patience: { type: 'integer', visible_if: { field: 'training.early_stopping_enabled', in: [true] } },
  prefetch: { type: 'integer', visible_if: { field: 'training.num_workers', gt: 0 } },
  prediction: { type: 'number', visible_if: { all: [{ field: 'method.prediction_branch', in: [true] }, {field:'training.training_objective',in:['reconstruction_prediction']}] } },
}};
describe('model training field visibility', () => {
  it('excludes irrelevant and invalid hidden fields', () => {
    expect(activeTrainingKeys(schema, {loss:'mse',ssim_weight:'invalid',num_workers:0}, {})).toEqual(['loss']);
    expect(activeTrainingKeys(schema, {loss:'ssim'}, {})).toEqual(['loss','ssim_window_size']);
    expect(activeTrainingKeys(schema, {loss:'mse_ssim'}, {})).toEqual(['loss','ssim_window_size','ssim_weight']);
  });
  it('honors model branches and option dependencies', () => {
    expect(activeTrainingKeys(schema, {early_stopping_enabled:true,num_workers:2,training_objective:'reconstruction_prediction'}, {prediction_branch:true})).toEqual(['loss','patience','prefetch','prediction']);
    expect(conditionMatches({any:[]},{},{})).toBe(false);
    expect(activeTrainingKeys(schema, {training_objective:'reconstruction_prediction'}, {prediction_branch:false})).toEqual(['loss']);
  });
});
