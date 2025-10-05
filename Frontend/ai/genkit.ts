
import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/google-genai';

// This is a server-only file.
// The 'ai' object is initialized with the API key from server-side environment variables.

export const ai = genkit({
  plugins: [
    googleAI({
      apiKey: process.env.GEMINI_API_KEY || '',
    }),
  ],
  logLevel: 'debug',
  enableTracingAndMetrics: true,
});
