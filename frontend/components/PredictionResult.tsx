"use client";

import type { PredictionResponse } from "@/lib/types";

const SEVERITY_STYLES: Record<string, string> = {
  "01-minor": "bg-green-100 text-green-800 border-green-300",
  "02-moderate": "bg-yellow-100 text-yellow-800 border-yellow-300",
  "03-severe": "bg-red-100 text-red-800 border-red-300",
  uncertain: "bg-gray-100 text-gray-700 border-gray-300",
};

const LABEL_MAP: Record<string, string> = {
  "01-minor": "Minor",
  "02-moderate": "Moderate",
  "03-severe": "Severe",
  uncertain: "Uncertain",
};

interface Props {
  result: PredictionResponse;
}

export default function PredictionResult({ result }: Props) {
  const style = SEVERITY_STYLES[result.prediction] ?? SEVERITY_STYLES.uncertain;
  const label = LABEL_MAP[result.prediction] ?? result.prediction;

  return (
    <div className="space-y-3">
      <div className={`inline-block border rounded-lg px-4 py-2 text-xl font-bold ${style}`}>
        {label}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
        <span>Confidence</span>
        <span className="font-medium text-gray-900">{(result.confidence * 100).toFixed(1)}%</span>
        <span>Model</span>
        <span className="font-medium text-gray-900">{result.model_version}</span>
        <span>Inference time</span>
        <span className="font-medium text-gray-900">{result.inference_time_ms.toFixed(1)} ms</span>
        {result.tta && (
          <>
            <span>TTA</span>
            <span className="font-medium text-blue-700">enabled (8 views)</span>
          </>
        )}
        {result.experiment && (
          <>
            <span>A/B Test</span>
            <span className="font-medium text-purple-700">
              {result.experiment} ({result.experiment_variant})
            </span>
          </>
        )}
      </div>
    </div>
  );
}
