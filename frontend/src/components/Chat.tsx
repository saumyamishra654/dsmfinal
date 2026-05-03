"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage, type ChatResponse } from "@/lib/api";
import ChatMessage from "./ChatMessage";

interface MessageEntry {
  question: string;
  response: ChatResponse;
}

const quickQueries = [
  "Show wireless subscriber growth over time",
  "Compare top 5 states by teledensity in 2022",
  "Plot UPI transaction growth since 2020",
  "What is the HHI trend before and after Jio?",
  "Show the convergence scatter plot for pre-Jio period",
];

export default function Chat() {
  const [apiKey, setApiKey] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem("dsm_api_key");
    if (saved) setApiKey(saved);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleApiKeyChange = (key: string) => {
    setApiKey(key);
    localStorage.setItem("dsm_api_key", key);
  };

  const handleSubmit = async (q: string) => {
    if (!q.trim()) return;
    setLoading(true);
    setQuery("");

    try {
      const response = await sendChatMessage(q, apiKey);
      setMessages((prev) => [...prev, { question: q, response }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          question: q,
          response: {
            type: "error",
            result: err instanceof Error ? err.message : "Unknown error occurred",
          },
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full not-prose">
      {/* API Key Input */}
      <div className="mb-6 p-4 bg-[#fef7ed] rounded-xl border border-[#e7d5c0]">
        <label className="block text-xs font-semibold text-[#78350f] mb-2 uppercase tracking-wide">
          API Key (stored in browser)
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => handleApiKeyChange(e.target.value)}
          placeholder="Enter your API key..."
          className="w-full px-4 py-2.5 border border-[#e7d5c0] rounded-lg text-sm bg-[#fdf8f3] focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none"
        />
      </div>

      {/* Quick queries */}
      <div className="mb-6">
        <p className="text-xs font-semibold text-[#78350f] mb-2 uppercase tracking-wide">
          Quick queries
        </p>
        <div className="flex flex-wrap gap-2">
          {quickQueries.map((q) => (
            <button
              key={q}
              onClick={() => handleSubmit(q)}
              disabled={loading}
              className="px-3 py-1.5 text-xs bg-[#fef7ed] border border-[#e7d5c0] rounded-full hover:bg-[#fed7aa] hover:border-amber-500 hover:text-[#9a3412] transition-colors disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto mb-6 space-y-4">
        {messages.length === 0 && !loading && (
          <div className="text-center py-16 text-gray-400">
            <div className="text-4xl mb-3">&#128202;</div>
            <p className="text-sm">
              Ask a question about the dataset to get started.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessage key={i} question={msg.question} response={msg.response} />
        ))}
        {loading && (
          <div className="flex items-center gap-3 p-5 bg-[#fef7ed] rounded-xl border border-[#e7d5c0]">
            <div className="w-5 h-5 border-2 border-amber-600 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-[#78350f]">Analyzing data...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="sticky bottom-0 bg-[#f5e6d3] pt-4 border-t border-[#e7d5c0]">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit(query);
          }}
          className="flex gap-3"
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about India's digital transformation data..."
            disabled={loading}
            className="flex-1 px-4 py-3 border border-[#e7d5c0] bg-[#fdf8f3] rounded-xl text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-3 bg-[#c2410c] text-white rounded-xl text-sm font-semibold hover:bg-[#9a3412] disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
