import { expect, it } from 'vitest';
import { SPATIAL_COLUMNS, SPATIAL_METRICS, SPATIAL_VERSION } from './metrics';

it('exposes the four metrics and their exact aggregate names', () => {
  expect(SPATIAL_VERSION).toBe('mad_median_q95_v2');
  expect(SPATIAL_METRICS.map(x => x.name)).toEqual(['D_med','R_med','D_q95','R_q95']);
  expect(SPATIAL_METRICS.map(x => x.aggregate)).toEqual(['D_agg_med','R_agg_med','D_agg_q95','R_agg_q95']);
  expect(SPATIAL_COLUMNS).toEqual(['D_med_in','D_med_out','Q_D_med','P_in_D_med','R_med_in','R_med_out','Q_R_med','D_q95_in','D_q95_out','Q_D_q95','P_in_D_q95','R_q95_in','R_q95_out','Q_R_q95']);
});
