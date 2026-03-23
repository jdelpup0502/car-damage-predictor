export interface PredictionResponse {
  model_version: string;
  prediction: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  inference_time_ms: number;
  uncertainty?: number;
  heatmap_png_base64?: string;
  tta?: boolean;
  experiment?: string;
  experiment_variant?: string;
}

export interface PredictOptions {
  heatmap: boolean;
  uncertainty: boolean;
  tta: boolean;
}
