import type { PredictionResponse, PredictOptions } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function predict(
  file: File,
  options: PredictOptions
): Promise<PredictionResponse> {
  const params = new URLSearchParams();
  if (options.heatmap) params.set("heatmap", "true");
  if (options.uncertainty) params.set("uncertainty", "true");
  if (options.tta) params.set("tta", "true");

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/predict?${params}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    if (res.status === 422) {
      const body = await res.json().catch(() => null);
      if (body?.detail?.error === "not_a_vehicle") {
        throw new Error("not_a_vehicle");
      }
    }
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}
