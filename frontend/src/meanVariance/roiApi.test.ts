import { afterEach, expect, it, vi } from 'vitest';
import { getVarianceRoiState, prepareVarianceRoi, evaluateVarianceRoi, listVarianceRoiJobs, abortVarianceRoiJob, deleteVarianceRoiJob, getVarianceRoiLog, varianceRoiImageUrl, varianceRoiArtifactUrl } from '../api';
afterEach(() => vi.unstubAllGlobals());
it('keeps ROI actions, editor layers and immutable downloads in the explicit project', async () => {
  const fetch = vi.fn().mockResolvedValue({ok:true,status:200,json:async()=>({})});
  vi.stubGlobal('fetch',fetch);
  const config = {roi: {x:3,y:5,width:7,height:9},opacity:.42};
  await getVarianceRoiState(8, 'ROI A');await prepareVarianceRoi(8,'ROI A');await evaluateVarianceRoi(8,config,'ROI A');
  await listVarianceRoiJobs('ROI A');await abortVarianceRoiJob(11,'ROI A');await getVarianceRoiLog(11,'ROI A');await deleteVarianceRoiJob(11,'ROI A');
  expect(fetch.mock.calls.every(([, options]) => options.headers['X-MLTrace-Project-ID'] === 'ROI A')).toBe(true);
  expect(fetch.mock.calls[2][0]).toBe('/api/mean-variance-analysis/runs/8/roi/evaluate');
  expect(JSON.parse(fetch.mock.calls[2][1].body)).toEqual(config);
  expect(varianceRoiImageUrl(8, 5, 'heatmap','ROI A')).toBe('/api/mean-variance-analysis/runs/8/roi/images/5/heatmap?project_id=ROI%20A');
  expect(varianceRoiArtifactUrl(8,11,'roi_table.png','ROI A',true)).toBe('/api/mean-variance-analysis/runs/8/roi/artifacts/11/roi_table.png?project_id=ROI%20A&download=true');
  expect(fetch.mock.calls.at(-1)?.[1].method).toBe('DELETE');
});
