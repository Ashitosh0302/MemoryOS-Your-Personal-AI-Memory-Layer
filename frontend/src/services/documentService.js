/**
 * documentService.js
 *
 * Thin wrapper around the MemoryOS documents API.
 * Uses VITE_API_BASE_URL from the .env file.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Upload a File object to the backend.
 *
 * @param {File} file
 * @returns {Promise<{
 *   success: boolean,
 *   filename: string,
 *   file_type: string,
 *   size: number,
 *   pages: number,
 *   text_length: number,
 *   message: string,
 * }>}
 * @throws {Error} with a user-friendly message on failure
 */
export async function uploadDocument(file) {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: 'POST',
    body: form,
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    // FastAPI validation errors come in data.detail
    const detail =
      typeof data.detail === 'string'
        ? data.detail
        : Array.isArray(data.detail)
        ? data.detail.map((e) => e.msg).join('; ')
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return data;
}
