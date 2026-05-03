export interface ChatResponse {
  type: "text" | "plotly" | "dataframe" | "error";
  result: string | Record<string, unknown>;
  code?: string;
}

export async function sendChatMessage(
  query: string,
  apiKey: string
): Promise<ChatResponse> {
  const baseUrl =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const response = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `API error (${response.status}): ${errorText || "Unknown error"}`
    );
  }

  return response.json();
}
