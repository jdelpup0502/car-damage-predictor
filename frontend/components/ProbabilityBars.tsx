"use client";

const BAR_COLORS: Record<string, string> = {
  "01-minor": "bg-green-500",
  "02-moderate": "bg-yellow-500",
  "03-severe": "bg-red-500",
};

const LABEL_MAP: Record<string, string> = {
  "01-minor": "Minor",
  "02-moderate": "Moderate",
  "03-severe": "Severe",
};

interface Props {
  probabilities: Record<string, number>;
}

export default function ProbabilityBars({ probabilities }: Props) {
  return (
    <div className="space-y-2">
      {Object.entries(probabilities).map(([cls, prob]) => (
        <div key={cls}>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-700">{LABEL_MAP[cls] ?? cls}</span>
            <span className="font-medium text-gray-900">{(prob * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${BAR_COLORS[cls] ?? "bg-blue-500"}`}
              style={{ width: `${(prob * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
