"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-96 bg-gray-50 rounded-xl animate-pulse" />,
}) as React.ComponentType<{
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  className?: string;
  useResizeHandler?: boolean;
  style?: React.CSSProperties;
}>;

interface InteractiveChartProps {
  dataFile: string;
  chartConfig: (data: Record<string, unknown>[]) => {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  };
  title?: string;
  height?: number;
}

export default function InteractiveChart({
  dataFile,
  chartConfig,
  title,
  height = 500,
}: InteractiveChartProps) {
  const [rawData, setRawData] = useState<Record<string, unknown>[] | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/data/${dataFile}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${dataFile}`);
        return res.json();
      })
      .then((data) => setRawData(data))
      .catch((err) => setError(err.message));
  }, [dataFile]);

  if (error) {
    return (
      <div className="my-8 p-6 rounded-xl border border-red-200 bg-red-50 text-red-700 text-sm">
        Error loading chart data: {error}
      </div>
    );
  }

  if (!rawData) {
    return (
      <div className="my-8 not-prose">
        <div
          className="rounded-xl border border-[#e7d5c0] bg-[#fef7ed] flex items-center justify-center"
          style={{ height }}
        >
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-gray-500 font-medium">
              Loading chart...
            </p>
          </div>
        </div>
      </div>
    );
  }

  const plotConfig = chartConfig(rawData);

  return (
    <div className="my-8 not-prose">
      {title && (
        <h4 className="text-base font-bold text-gray-800 mb-3 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-500" />
          {title}
        </h4>
      )}
      <div className="rounded-xl border border-gray-200 overflow-hidden shadow-lg bg-white">
        <Plot
          data={plotConfig.data}
          layout={{
            height,
            margin: { t: 50, r: 40, b: 60, l: 70 },
            font: { family: "Inter, system-ui, sans-serif", size: 12 },
            paper_bgcolor: "white",
            plot_bgcolor: "#fafbfc",
            ...plotConfig.layout,
            xaxis: {
              gridcolor: "#f1f5f9",
              ...(plotConfig.layout.xaxis as object),
            },
            yaxis: {
              gridcolor: "#f1f5f9",
              ...(plotConfig.layout.yaxis as object),
            },
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
          }}
          className="w-full"
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}
