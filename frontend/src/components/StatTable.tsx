interface StatRow {
  label: string;
  value: string;
  significant?: boolean;
}

interface StatTableProps {
  title: string;
  rows: StatRow[];
}

export default function StatTable({ title, rows }: StatTableProps) {
  return (
    <div className="my-8 not-prose">
      <div className="rounded-xl border border-gray-200 overflow-hidden shadow-sm">
        <div className="bg-[#fef7ed] px-5 py-3 border-b border-[#e7d5c0]">
          <h4 className="text-sm font-bold text-gray-800 uppercase tracking-wide">
            {title}
          </h4>
        </div>
        <table className="w-full text-sm">
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className={`border-b border-gray-100 last:border-0 ${
                  row.significant
                    ? "bg-blue-50 hover:bg-blue-100"
                    : "hover:bg-gray-50"
                }`}
              >
                <td className="px-5 py-3 font-medium text-gray-700">
                  {row.label}
                </td>
                <td
                  className={`px-5 py-3 text-right font-mono ${
                    row.significant
                      ? "text-blue-700 font-bold"
                      : "text-gray-900"
                  }`}
                >
                  {row.value}
                  {row.significant && (
                    <span className="ml-2 text-xs bg-blue-200 text-blue-800 px-1.5 py-0.5 rounded-full font-sans font-semibold">
                      sig
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
