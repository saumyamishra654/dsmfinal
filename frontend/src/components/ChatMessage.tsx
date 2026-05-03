"use client";

import { useState } from "react";
import type { ChatResponse } from "@/lib/api";
import PlotlyChart from "./PlotlyChart";
import DataTable from "./DataTable";

interface ChatMessageProps {
  question: string;
  response: ChatResponse;
}

export default function ChatMessage({ question, response }: ChatMessageProps) {
  const [showCode, setShowCode] = useState(false);

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden shadow-sm my-4">
      {/* Question */}
      <div className="bg-slate-50 px-5 py-3 border-b border-gray-200">
        <p className="text-sm font-semibold text-gray-800 flex items-center gap-2">
          <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-xs font-bold">
            Q
          </span>
          {question}
        </p>
      </div>

      {/* Response */}
      <div className="p-5">
        {response.type === "plotly" && (
          <PlotlyChart
            data={(response.result as Record<string, unknown>)?.data as Plotly.Data[] || []}
            layout={(response.result as Record<string, unknown>)?.layout as Partial<Plotly.Layout> || {}}
          />
        )}

        {response.type === "dataframe" && (
          <DataTable data={response.result as unknown as Record<string, unknown>[]} />
        )}

        {response.type === "text" && (
          <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
            {String(response.result)}
          </p>
        )}

        {response.type === "error" && (
          <pre className="text-sm text-red-700 bg-red-50 p-4 rounded-lg overflow-x-auto border border-red-200">
            {String(response.result)}
          </pre>
        )}
      </div>

      {/* Show code toggle */}
      {response.code && (
        <div className="border-t border-gray-200">
          <button
            onClick={() => setShowCode(!showCode)}
            className="w-full px-5 py-2.5 text-xs font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50 text-left flex items-center gap-2"
          >
            <svg
              className={`w-3 h-3 transition-transform ${showCode ? "rotate-90" : ""}`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z"
                clipRule="evenodd"
              />
            </svg>
            {showCode ? "Hide code" : "Show code"}
          </button>
          {showCode && (
            <pre className="px-5 pb-4 text-xs font-mono text-gray-700 bg-gray-50 overflow-x-auto">
              {response.code}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
