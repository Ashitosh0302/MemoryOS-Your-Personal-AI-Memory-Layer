/**
 * ragService.js
 *
 * Thin wrapper around the MemoryOS RAG API.
 * Uses VITE_API_BASE_URL from the .env file.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Send a natural-language question to the RAG endpoint.
 *
 * @param {string} query  The user's question
 * @param {number} [top_k=5] Number of source chunks to retrieve
 * @returns {Promise<{
 *   success: boolean,
 *   query: string,
 *   answer: string,
 *   sources: Array<{
 *     id: string,
 *     text: string,
 *     metadata: Record<string, any>,
 *     distance?: number
 *   }>,
 *   total_sources: number
 * }>}
 * @throws {Error} User-friendly error message on failure
 */
export async function askMemory(query, top_k = 5) {
  if (!query || !query.trim()) {
    throw new Error('Please enter a question.');
  }

  const response = await fetch(`${API_BASE}/api/rag`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query.trim(),
      top_k,
    }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      typeof data.detail === 'string'
        ? data.detail
        : Array.isArray(data.detail)
        ? data.detail.map((e) => e.msg).join('; ')
        : `HTTP ${response.status}: Request failed`;
    throw new Error(detail);
  }

  return data;
}
