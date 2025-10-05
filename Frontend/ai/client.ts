'use client';
// This file contains client-safe code for interacting with the AI setup.
export const KEY_NAME = 'gemini_api_key';

export function getApiKey(): string | null {
  if (typeof window !== 'undefined') {
    return window.localStorage.getItem(KEY_NAME);
  }
  return null;
}

export function saveApiKey(apiKey: string) {
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(KEY_NAME, apiKey);
  }
}
