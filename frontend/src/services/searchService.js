/**
 * searchService.js
 *
 * Thin wrapper around the MemoryOS semantic search API.
 * Uses VITE_API_BASE_URL from the .env file.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Perform a semantic search query against the MemoryOS vector store.
 *
 * @param {string} query Natural language search query
 * @param {number} [top_k=5] Number of top results to return
 * @returns {Promise<{
 *   success: boolean,
 *   query: string,
 *   total_results: number,
 *   results: Array<{
 *     id: string,
 *     text: string,
 *     metadata: Record<string, any>,
 *     distance?: number
 *   }>
 * }>}
 * @throws {Error} User-friendly error message on failure
 */
export async function searchMemory(query, top_k = 5) {
  if (!query || !query.trim()) {
    throw new Error('Please enter a question or query string.');
  }

  const response = await fetch(`${API_BASE}/api/search`, {
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
        : `HTTP ${response.status}: Search request failed`;
    throw new Error(detail);
  }

  return data;
}
