"use client";

import type { PredictOptions } from "@/lib/types";

interface Props {
  options: PredictOptions;
  onChange: (options: PredictOptions) => void;
}

const OPTIONS: { key: keyof PredictOptions; label: string; description: string }[] = [
  { key: "heatmap", label: "Grad-CAM Heatmap", description: "Show which regions influenced the prediction" },
  { key: "uncertainty", label: "Uncertainty Score", description: "Predictive entropy (0 = certain, 1 = max uncertain)" },
  { key: "tta", label: "Test-Time Augmentation", description: "Average 8 augmented views (~8x slower)" },
];

export default function SettingsPanel({ options, onChange }: Props) {
  function toggle(key: keyof PredictOptions) {
    onChange({ ...options, [key]: !options[key] });
  }

  return (
    <div className="space-y-2">
      {OPTIONS.map(({ key, label, description }) => (
        <label key={key} className="flex items-start gap-3 cursor-pointer group">
          <input
            type="checkbox"
            checked={options[key]}
            onChange={() => toggle(key)}
            className="mt-0.5 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <div>
            <p className="text-sm font-medium text-gray-800 group-hover:text-blue-700">{label}</p>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
        </label>
      ))}
    </div>
  );
}
