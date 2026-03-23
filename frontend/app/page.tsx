"use client";

import { useState } from "react";
import ImageUploader from "@/components/ImageUploader";
import PredictionResult from "@/components/PredictionResult";
import ProbabilityBars from "@/components/ProbabilityBars";
import HeatmapViewer from "@/components/HeatmapViewer";
import UncertaintyBadge from "@/components/UncertaintyBadge";
import SettingsPanel from "@/components/SettingsPanel";
import { predict } from "@/lib/api";
import type { PredictionResponse, PredictOptions } from "@/lib/types";

export default function Home() {
  const [options, setOptions] = useState<PredictOptions>({
    heatmap: false,
    uncertainty: false,
    tta: false,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);

  async function runPrediction(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predict(file, options);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file: File) {
    setCurrentFile(file);
    runPrediction(file);
  }

  function handleReanalyze() {
    if (currentFile) runPrediction(currentFile);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Car Damage Severity Predictor</h1>
        <p className="text-gray-500 mt-1">Upload a photo of car damage to get an AI-powered severity assessment.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left column: upload + settings */}
        <div className="space-y-6">
          <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-semibold mb-4">Upload Image</h2>
            <ImageUploader onFile={handleFile} loading={loading} />
            {loading && (
              <p className="text-sm text-blue-600 mt-3 animate-pulse">Analyzing image...</p>
            )}
            {error && (
              <p className="text-sm text-red-600 mt-3">Error: {error}</p>
            )}
          </section>

          <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h2 className="text-lg font-semibold mb-4">Options</h2>
            <SettingsPanel options={options} onChange={setOptions} />
            {currentFile && (
              <button
                onClick={handleReanalyze}
                disabled={loading}
                className="mt-5 w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-medium rounded-lg transition-colors"
              >
                {loading ? "Analyzing..." : "Re-analyze"}
              </button>
            )}
          </section>
        </div>

        {/* Right column: results */}
        <div className="space-y-6">
          {result ? (
            <>
              <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="text-lg font-semibold mb-4">Prediction</h2>
                <PredictionResult result={result} />
                {result.uncertainty !== undefined && (
                  <div className="mt-4">
                    <UncertaintyBadge value={result.uncertainty} />
                  </div>
                )}
              </section>

              <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="text-lg font-semibold mb-4">Class Probabilities</h2>
                <ProbabilityBars probabilities={result.all_probabilities} />
              </section>

              {result.heatmap_png_base64 && (
                <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                  <HeatmapViewer base64={result.heatmap_png_base64} />
                </section>
              )}
            </>
          ) : (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex items-center justify-center h-48 text-gray-400">
              <p>Results will appear here after upload.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
