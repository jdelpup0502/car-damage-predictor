"use client";

interface Props {
  value: number;
}

function getStyle(v: number) {
  if (v < 0.3) return "bg-green-100 text-green-800";
  if (v < 0.6) return "bg-yellow-100 text-yellow-800";
  return "bg-red-100 text-red-800";
}

function getLabel(v: number) {
  if (v < 0.3) return "Low";
  if (v < 0.6) return "Medium";
  return "High";
}

export default function UncertaintyBadge({ value }: Props) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-600">Uncertainty</span>
      <span className={`text-sm font-medium px-2 py-0.5 rounded-full ${getStyle(value)}`}>
        {getLabel(value)} ({(value * 100).toFixed(1)}%)
      </span>
    </div>
  );
}
