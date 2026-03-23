"use client";

interface Props {
  base64: string;
}

export default function HeatmapViewer({ base64 }: Props) {
  return (
    <div>
      <p className="text-sm font-medium text-gray-700 mb-2">Grad-CAM Heatmap</p>
      <img
        src={`data:image/png;base64,${base64}`}
        alt="Grad-CAM heatmap"
        className="rounded-lg w-full max-w-xs object-contain border border-gray-200"
      />
      <p className="text-xs text-gray-400 mt-1">Highlights regions influencing the prediction</p>
    </div>
  );
}
