"use client";

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

interface PlotlyChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
}

export default function PlotlyChart({
  data,
  layout = {},
  height = 450,
}: PlotlyChartProps) {
  return (
    <div className="rounded-xl border border-gray-200 overflow-hidden shadow-sm bg-white my-4">
      <Plot
        data={data}
        layout={{
          height,
          margin: { t: 40, r: 30, b: 50, l: 60 },
          font: { family: "Inter, system-ui, sans-serif", size: 12 },
          paper_bgcolor: "white",
          plot_bgcolor: "#fafbfc",
          ...layout,
          xaxis: { gridcolor: "#f1f5f9", ...(layout.xaxis as object) },
          yaxis: { gridcolor: "#f1f5f9", ...(layout.yaxis as object) },
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d"],
        }}
        useResizeHandler
        style={{ width: "100%" }}
        className="w-full"
      />
    </div>
  );
}
