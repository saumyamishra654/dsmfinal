interface DataTableProps {
  data: Record<string, unknown>[];
  maxRows?: number;
}

export default function DataTable({ data, maxRows = 50 }: DataTableProps) {
  if (!data || data.length === 0) return null;

  const columns = Object.keys(data[0]);
  const displayData = data.slice(0, maxRows);

  return (
    <div className="my-6 not-prose rounded-xl border border-gray-200 overflow-hidden shadow-sm">
      <div className="overflow-x-auto max-h-96">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col} className="whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col} className="whitespace-nowrap font-mono text-xs">
                    {row[col] != null ? String(row[col]) : "-"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > maxRows && (
        <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
          Showing {maxRows} of {data.length} rows
        </div>
      )}
    </div>
  );
}
