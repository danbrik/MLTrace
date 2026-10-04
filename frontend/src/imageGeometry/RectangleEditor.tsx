import { Alert, Loader, Stack, Text } from '@mantine/core';
import { useRef, useState, type PointerEvent, type KeyboardEvent } from 'react';
import { dragRectangle, localToImage, normalizeAngle, roiCorners, rotateRectangle, type Corner, type DragMode, type RotatedRoiRectangle } from './rectangleGeometry';

export function RectangleEditor({width, height, imageSrc, imageAlt, value, disabled = false, invalid = false, onChange, label = 'ROI', hint = '', testId = 'roi-editor'}: {
  width: number; height: number; imageSrc?: string; imageAlt: string;
  value: RotatedRoiRectangle; disabled?: boolean; invalid?: boolean; onChange: (next: RotatedRoiRectangle) => void;
  label?: string; hint?: string; testId?: string;
}) {
  const name = label;
  const container = useRef<HTMLDivElement>(null);
  const drag = useRef<{mode: DragMode; x: number; y: number; rect: RotatedRoiRectangle} | null>(null);
  const [loaded, setLoaded] = useState('');
  const [failed, setFailed] = useState('');
  const error = !!imageSrc && failed === imageSrc;
  const loading = !error && (!imageSrc || loaded !== imageSrc);
  const locked = disabled || loading || error;
  const color = invalid ? '#e03131' : '#ff9800';
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
    onChange(previous.mode === 'rotate' ? rotateRectangle(previous.rect, previous, next)
      : dragRectangle(previous.rect, previous.mode, next.x - previous.x, next.y - previous.y));
  }
  function key(event: KeyboardEvent, mode: DragMode) {
    if (locked || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    if (mode === 'rotate') {
      onChange({...value, angle_degrees: normalizeAngle(value.angle_degrees + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -step : step))});
    } else {
      onChange(dragRectangle(value, mode, event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0,
        event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0));
    }
  }
  const points = roiCorners(value);
  const corners: [Corner, string][] = [['tl','oben links'],['tr','oben rechts'],['br','unten rechts'],['bl','unten links']];
  // An inset rotation handle stays usable even for the initial full-image ROI.
  const rotationHandle = localToImage(value, 0, -value.height / 2 + Math.min(value.height / 4, height * .07));
  return <Stack gap="xs">
    <Text size="sm">Ecken ziehen, das Rechteck verschieben oder am runden Griff drehen. Pfeiltasten: 1 Pixel beziehungsweise 1°, mit Umschalt: 10. {hint}</Text>
    {error && <Alert color="red">Das Eingangsbild konnte nicht angezeigt werden.</Alert>}
    {loading && <Text size="sm" role="status"><Loader size="xs" /> Vorschau wird aktualisiert …</Text>}
    <div style={{padding: 12, overflow: 'hidden'}}>
      <div ref={container} data-testid={testId} style={{position: 'relative', width: '100%', aspectRatio: `${width}/${height}`, touchAction: 'none'}}
        onPointerMove={move} onPointerUp={() => {drag.current = null;}} onPointerCancel={() => {drag.current = null;}} onLostPointerCapture={() => {drag.current = null;}}>
        {imageSrc && <img key={imageSrc} alt={imageAlt} src={imageSrc} draggable={false}
          onLoad={() => setLoaded(imageSrc)} onError={() => setFailed(imageSrc)}
          style={{display: 'block', width: '100%', height: '100%', imageRendering: 'pixelated', visibility: loading ? 'hidden' : 'visible'}} />}
        <svg viewBox={`0 0 ${width} ${height}`} style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none'}}>
          <polygon role="button" tabIndex={locked ? -1 : 0} aria-label={`${label} verschieben`} aria-disabled={locked} points={points.map(p => `${p.x},${p.y}`).join(' ')}
            fill="transparent" stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" style={{pointerEvents:'all', cursor: locked ? 'default' : 'move'}}
            onPointerDown={event => start(event, 'move')} onKeyDown={event => key(event, 'move')} />
          <line x1={(points[0].x + points[1].x)/2} y1={(points[0].y + points[1].y)/2} x2={rotationHandle.x} y2={rotationHandle.y} stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </svg>
        {corners.map(([corner, label], i) => <button key={corner} type="button" aria-label={`${name} Ecke ${label}`} disabled={locked}
          onPointerDown={event => start(event, corner)} onKeyDown={event => key(event, corner)}
          style={{position: 'absolute', left: `${points[i].x / width * 100}%`, top: `${points[i].y / height * 100}%`, transform: 'translate(-50%, -50%)', width: 16, height: 16, padding: 0,
            border: '2px solid white', borderRadius: 3, background: color, cursor: corner === 'tl' || corner === 'br' ? 'nwse-resize' : 'nesw-resize'}} />)}
        <button type="button" aria-label={`${label} drehen`} title={`${label} drehen`} disabled={locked} onPointerDown={event => start(event, 'rotate')} onKeyDown={event => key(event, 'rotate')}
          style={{position: 'absolute', left: `${rotationHandle.x / width * 100}%`, top: `${rotationHandle.y / height * 100}%`, transform: 'translate(-50%, -50%)', width: 24, height: 24, padding: 0,
            border: '2px solid white', borderRadius: '50%', background: color, color: 'black', cursor: 'grab'}}>↻</button>
      </div>
    </div>
  </Stack>;
}
