import { Alert, Button, Group, NumberInput, Stack, Text } from '@mantine/core';
import { RectangleEditor } from '../../imageGeometry/RectangleEditor';
import { roiCoordinates, roiValidation } from '../../imageGeometry/rectangleGeometry';
import { cropRectangle } from './cropGeometry';
import type { StepControl, StepControlProps } from './types';

function CropBoxControl({ inputImage, config, disabled, onChange }: StepControlProps) {
  const rect = cropRectangle(config, inputImage);
  const error = roiValidation(rect, inputImage.width, inputImage.height)?.replaceAll('ROI', 'Crop');
  const finite = Object.values(rect).every(Number.isFinite);
  return <Stack gap="xs">
    <Group align="end">
      <NumberInput label="Winkel (°)" description="Positive Winkel drehen im Uhrzeigersinn." step={1}
        value={Number.isFinite(rect.angle_degrees) ? rect.angle_degrees : ''} disabled={disabled}
        onChange={value => onChange({roi: {...rect, angle_degrees: value === '' ? NaN : Number(value)}})} />
      <Button variant="light" disabled={disabled} onClick={() => onChange({roi: {...rect, angle_degrees: 0}})}>Drehung zurücksetzen</Button>
    </Group>
    {rect.width === inputImage.width && rect.height === inputImage.height && <Text size="sm" c="dimmed">Vor dem Drehen die Vollbildauswahl verkleinern, damit alle Ecken innerhalb des Bildes bleiben.</Text>}
    <Text size="sm" c="dimmed">Der Ausschnitt wird mit Nearest Neighbor gerade ausgerichtet, ohne Glättung oder Mischung benachbarter Pixelwerte. Eine optionale Größenanpassung erfolgt danach.</Text>
    {error && <Alert color="red">{error}</Alert>}
    {finite && <RectangleEditor width={inputImage.width} height={inputImage.height} imageSrc={inputImage.image_data_url}
      imageAlt="Crop-Eingangsbild" value={rect} disabled={disabled} invalid={!!error} label="Crop" testId="crop-editor"
      onChange={roi => onChange({roi})} />}
    {finite && <Text size="xs">{roiCoordinates(rect)}</Text>}
  </Stack>;
}

export const cropBoxControl: StepControl = {
  component: CropBoxControl,
  ownedKeys: ['x', 'y', 'width', 'height', 'roi'],
};
