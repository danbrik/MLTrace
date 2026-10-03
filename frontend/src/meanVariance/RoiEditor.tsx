import { Alert, Stack, Text } from '@mantine/core';
import { useRef, useState, type PointerEvent, type KeyboardEvent } from 'react';
import { dragRectangle, type Corner, type DragMode } from './roiGeometry';
import type { RoiRectangle } from './roiTypes';

export function RoiEditor({width, height, background, heatmap, opacity, value, disabled, onChange}: {
  width: number; height: number; background: string; heatmap: string; opacity: number;
  value: RoiRectangle; disabled: boolean; onChange: (next: RoiRectangle) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const drag = useRef<{mode: DragMode; x: number; y: number; rect: RoiRectangle} | null>(null);
  const [loaded, setLoaded] = useState({background: false, heatmap: false});
  const [error, setError] = useState(false);
  const locked = disabled || !loaded.background || !loaded.heatmap || error;
  function point(event: PointerEvent) {
    const bounds = container.current!.getBoundingClientRect();
    return {x: (event.clientX - bounds.left) / bounds.width * width, y: (event.clientY - bounds.top) / bounds.height * height};
  }
  function start(event: PointerEvent, mode: DragMode) {
    if (locked) return;
    event.preventDefault(); event.stopPropagation();
    drag.current = {...point(event), rect: {...value}, mode};
    container.current!.setPointerCapture(event.pointerId);
  }
  function move(event: PointerEvent) {
    if (locked || !drag.current) return;
    const next = point(event), previous = drag.current;
    onChange(dragRectangle(previous.rect, previous.mode, next.x - previous.x, next.y - previous.y, width, height));
  }
  function key(event: KeyboardEvent, mode: DragMode) {
    if (locked || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    onChange(dragRectangle(value, mode, event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0,
      event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0, width, height));
  }
  const corners: [Corner, number, number, string][] = [
    ['tl', value.x, value.y, 'oben links'], ['tr', value.x + value.width, value.y, 'oben rechts'],
    ['bl', value.x, value.y + value.height, 'unten links'], ['br', value.x + value.width, value.y + value.height, 'unten rechts'],
  ];
  return <Stack gap="xs">
    <Text size="sm">Ecken ziehen oder das Rechteck verschieben. Mit den Pfeiltasten: 1 Pixel, mit Umschalt: 10 Pixel. Die ROI gilt für alle Paare.</Text>
    {error && <Alert color="red">Die Editorbilder konnten nicht geladen werden. Bitte die ROI-Auswertung erneut öffnen.</Alert>}
    <div style={{padding: 12}}>
      <div ref={container} data-testid="roi-editor" style={{position: 'relative', width: '100%', aspectRatio: `${width}/${height}`, touchAction: 'none'}}
        onPointerMove={move} onPointerUp={() => {drag.current = null;}} onPointerCancel={() => {drag.current = null;}} onLostPointerCapture={() => {drag.current = null;}}>
        <img alt="Mittelbild der Normalphase" src={background} draggable={false} onLoad={() => setLoaded(v => ({...v, background: true}))} onError={() => setError(true)}
          style={{display: 'block', width: '100%', height: '100%', imageRendering: 'pixelated'}} />
        <img alt="Varianzdifferenz als transparente Heatmap" src={heatmap} draggable={false} onLoad={() => setLoaded(v => ({...v, heatmap: true}))} onError={() => setError(true)}
          style={{position: 'absolute', inset: 0, width: '100%', height: '100%', opacity, pointerEvents: 'none', imageRendering: 'pixelated'}} />
        <div role="button" tabIndex={locked ? -1 : 0} aria-label="ROI verschieben" aria-disabled={locked} onPointerDown={event => start(event, 'move')} onKeyDown={event => key(event, 'move')}
          style={{position: 'absolute', left: `${value.x / width * 100}%`, top: `${value.y / height * 100}%`, width: `${value.width / width * 100}%`, height: `${value.height / height * 100}%`,
            border: '2px solid #ff9800', boxSizing: 'border-box', cursor: locked ? 'default' : 'move'}} />
        {corners.map(([corner, x, y, label]) => <button key={corner} type="button" aria-label={`ROI Ecke ${label}`} disabled={locked}
          onPointerDown={event => start(event, corner)} onKeyDown={event => key(event, corner)}
          style={{position: 'absolute', left: `${x / width * 100}%`, top: `${y / height * 100}%`, transform: 'translate(-50%, -50%)', width: 16, height: 16, padding: 0,
            border: '2px solid white', borderRadius: 3, background: '#ff9800', cursor: corner === 'tl' || corner === 'br' ? 'nwse-resize' : 'nesw-resize'}} />)}
      </div>
    </div>
  </Stack>;
}
