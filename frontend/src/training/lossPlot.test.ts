import { describe, it, expect } from 'vitest';
import { lossMetricsCsv, lossPlotRevision, supportsLossPlot } from './lossPlot';
import type { TrainingRun } from '../types';
describe('loss exports', () => {
  it('preserves precision, sorts epochs and leaves missing values empty', () => {
    expect(lossMetricsCsv([{epoch:2,train_loss:.123456789012345,val_loss:null},{epoch:1,train_loss:Infinity,val_loss:1}]))
      .toBe('epoch,train_loss,val_loss\r\n1,,1\r\n2,0.123456789012345,\r\n');
  });
  it('supports only epoch-based AE/VAE builders', () => {
    for (const builder_kind of ['sequential_autoencoder','sequential_spatial_autoencoder','sequential_variational_autoencoder','spatiotemporal_autoencoder']) {
      expect(supportsLossPlot({builder_kind,training_mode:'gradient'})).toBe(true);
    }
    expect(supportsLossPlot({builder_kind:'fast_anogan',training_mode:'gradient'})).toBe(false);
    expect(supportsLossPlot({builder_kind:'form',training_mode:'fit'})).toBe(false);
  });
  it('updates for metrics and restart but not unrelated status updates', () => {
    const run = { enqueued_at:'a', started_at:'b',metrics:[{epoch:1,train_loss:1,val_loss:null}] } as TrainingRun;
    expect(lossPlotRevision({...run,status:'finished'})).toBe(lossPlotRevision(run));
    expect(lossPlotRevision({...run,metrics:[]})).not.toBe(lossPlotRevision(run));
    expect(lossPlotRevision({...run,enqueued_at:'c'})).not.toBe(lossPlotRevision(run));
  });
});
