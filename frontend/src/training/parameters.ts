import type { ConfigSchema, TrainingCondition } from '../types';
export function conditionMatches(condition: TrainingCondition | undefined, training: Record<string, unknown>, method: Record<string, unknown>): boolean {
  if (!condition) return true;
  if ('all' in condition) return condition.all.every(item => conditionMatches(item, training, method));
  if ('any' in condition) return condition.any.some(item => conditionMatches(item, training, method));
  const [source, key] = condition.field.split('.');
  const value = (source === 'method' ? method : training)[key];
  return condition.in ? condition.in.includes(value) : typeof value === 'number' && value > (condition.gt ?? 0);
}
export function activeTrainingKeys(schema: ConfigSchema | undefined, training: Record<string, unknown>, method: Record<string, unknown>): string[] {
  return Object.entries(schema?.properties ?? {}).filter(([, property]) => conditionMatches(property.visible_if, training, method)).map(([key]) => key);
}
