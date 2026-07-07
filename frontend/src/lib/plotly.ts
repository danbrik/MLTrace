// Schlankes Plotly-Bundle: nur die Trace-Typen, die Analysis braucht
// (scatter fuer Zeitreihen, heatmap fuer die Fehler-Matrix). Original/
// Reconstruction-Panels und Video-Frames sind normale <img>-Tags, daher kein
// image-Trace (der zieht ein 'buffer/'-Polyfill rein und bricht optimizeDeps).
// Vermeidet ausserdem das ~3.5MB grosse plotly.js-dist.
import Plotly from 'plotly.js/lib/core';
import scatter from 'plotly.js/lib/scatter';
import heatmap from 'plotly.js/lib/heatmap';
import bar from 'plotly.js/lib/bar';

import type { Data, Layout, Config, PlotlyHTMLElement } from 'plotly.js';

Plotly.register([scatter, heatmap, bar]);

export type { Data, Layout, Config, PlotlyHTMLElement };
export default Plotly as typeof import('plotly.js');
