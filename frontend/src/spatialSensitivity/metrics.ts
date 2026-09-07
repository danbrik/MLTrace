export const SPATIAL_VERSION = 'mad_median_q95_v2' as const;
export const SPATIAL_METRICS = [
  {name:'D_med', title:'Absolute change (median)', aggregate:'D_agg_med'},
  {name:'R_med', title:'Robust normalized change (median)', aggregate:'R_agg_med'},
  {name:'D_q95', title:'Absolute change (Q95)', aggregate:'D_agg_q95'},
  {name:'R_q95', title:'Robust normalized change (Q95)', aggregate:'R_agg_q95'},
] as const;
export const SPATIAL_COLUMNS = SPATIAL_METRICS.flatMap(({name}) => [
  `${name}_in`, `${name}_out`, `Q_${name}`, ...(name.startsWith('D_') ? [`P_in_${name}`] : []),
]);
