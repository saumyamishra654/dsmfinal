import Chat from "@/components/Chat";

export default function ExplorerPage() {
  return (
    <div className="min-h-[calc(100vh-5rem)]">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2 text-2xl font-bold">
          Data Explorer
        </h1>
        <p className="text-slate-300 text-sm">
          Ask questions about India&apos;s digital transformation data using
          natural language. The AI assistant can generate charts, tables, and
          statistical analyses on demand.
        </p>
      </div>

      <div className="prose mb-6">
        <div className="callout callout-blue">
          <strong>How it works:</strong> Type a question in natural language
          below. The backend will generate Python code to query the dataset and
          return visualizations, tables, or text answers. You can ask about
          trends, comparisons, statistical tests, or request specific
          visualizations.
        </div>
      </div>

      <Chat />
    </div>
  );
}
