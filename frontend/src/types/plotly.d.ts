declare module "react-plotly.js" {
  import { Component } from "react";

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    frames?: Plotly.Frame[];
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }, graphDiv: HTMLElement) => void;
    onPurge?: (figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}

declare namespace Plotly {
  interface Data {
    x?: (string | number | Date)[];
    y?: (string | number | Date)[];
    z?: (string | number | Date)[][] | (string | number | Date)[];
    type?: string;
    mode?: string;
    name?: string;
    text?: string | string[];
    textposition?: string;
    textfont?: { size?: number; color?: string; family?: string };
    marker?: {
      size?: number | number[];
      color?: string | number[] | string[];
      colorscale?: string;
      showscale?: boolean;
      colorbar?: { title?: string; thickness?: number };
      line?: { width?: number; color?: string };
      symbol?: string;
    };
    line?: {
      color?: string;
      width?: number;
      dash?: string;
      shape?: string;
    };
    fill?: string;
    fillcolor?: string;
    hoverinfo?: string;
    hovertemplate?: string;
    [key: string]: unknown;
  }

  interface Layout {
    title?: string | { text?: string; font?: { size?: number; family?: string; color?: string } };
    height?: number;
    width?: number;
    margin?: { t?: number; r?: number; b?: number; l?: number; pad?: number };
    font?: { family?: string; size?: number; color?: string };
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    xaxis?: Axis;
    yaxis?: Axis;
    showlegend?: boolean;
    legend?: { x?: number; y?: number; bgcolor?: string; font?: { size?: number } };
    annotations?: Annotation[];
    shapes?: Shape[];
    [key: string]: unknown;
  }

  interface Axis {
    title?: string | { text?: string };
    type?: string;
    gridcolor?: string;
    range?: [number, number];
    dtick?: number;
    tickformat?: string;
    showgrid?: boolean;
    zeroline?: boolean;
    [key: string]: unknown;
  }

  interface Annotation {
    x?: string | number;
    y?: string | number;
    xref?: string;
    yref?: string;
    text?: string;
    showarrow?: boolean;
    arrowhead?: number;
    ax?: number;
    ay?: number;
    font?: { color?: string; size?: number; family?: string };
    [key: string]: unknown;
  }

  interface Shape {
    type?: string;
    x0?: string | number;
    y0?: string | number;
    x1?: string | number;
    y1?: string | number;
    line?: { color?: string; width?: number; dash?: string };
    [key: string]: unknown;
  }

  interface Config {
    responsive?: boolean;
    displayModeBar?: boolean;
    displaylogo?: boolean;
    modeBarButtonsToRemove?: string[];
    [key: string]: unknown;
  }

  interface Frame {
    name?: string;
    data?: Data[];
    layout?: Partial<Layout>;
    [key: string]: unknown;
  }
}
