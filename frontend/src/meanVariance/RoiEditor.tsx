import { Alert } from '@mantine/core';
import { useEffect, useState } from 'react';
import { requestRoiPreview } from './roiPreview';
import { RectangleEditor } from '../imageGeometry/RectangleEditor';
import type { RotatedRoiRectangle } from './roiTypes';

export function RoiEditor({width, height, previewUrl, value, disabled, invalid, onChange}: {
  width: number; height: number; previewUrl: string;
  value: RotatedRoiRectangle; disabled: boolean; invalid: boolean; onChange: (next: RotatedRoiRectangle) => void;
}) {
  const [preview, setPreview] = useState<{url: string; src?: string; error?: string} | null>(null);
  useEffect(() => {
    setPreview(null);
    let objectUrl: string | undefined;
    const cancel = requestRoiPreview(previewUrl, blob => {
      objectUrl = URL.createObjectURL(blob);
      setPreview({url: previewUrl, src: objectUrl});
    }, error => setPreview({url: previewUrl, error}));
    return () => {cancel(); if (objectUrl) URL.revokeObjectURL(objectUrl);};
  }, [previewUrl]);
  const current = preview?.url === previewUrl ? preview : null;
  return <>
    {current?.error && <Alert color="red">{current.error}</Alert>}
    <RectangleEditor width={width} height={height} imageSrc={current?.src}
      imageAlt="Normalmittelbild mit örtlicher Varianz-Heatmap" value={value} disabled={disabled || !!current?.error}
      invalid={invalid} onChange={onChange} hint="Die ROI gilt für alle Paare." />
  </>;
}
