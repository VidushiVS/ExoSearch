'use server';
/**
 * @fileoverview A flow for generating a visual representation of an exoplanet.
 *
 * This file exports:
 * - `generatePlanetVisual`: A function to generate an image of an exoplanet based on its data.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import type {Exoplanet} from '@/lib/types';
import {
  GeneratePlanetVisualInputSchema,
  type GeneratePlanetVisualInput,
} from './types';

const imageGenPrompt = ai.definePrompt({
  name: 'planetImagePrompt',
  input: {
    schema: GeneratePlanetVisualInputSchema,
  },
  output: {
    format: 'media',
  },
  prompt: `{{prompt}}`,
});

const generatePlanetVisualFlow = ai.defineFlow(
  {
    name: 'generatePlanetVisualFlow',
    inputSchema: GeneratePlanetVisualInputSchema,
    outputSchema: z.object({
      imageUrl: z.string(),
      description: z.string(),
    }),
  },
  async (input: {planet: Exoplanet; prompt: string}) => {
    const {media} = await ai.generate({
      prompt: input.prompt,
      model: 'googleai/gemini-2.5-flash-image-preview',
      config: {
        responseModalities: ['IMAGE'],
      },
    });

    if (!media?.url) {
      throw new Error('Image generation failed to return a valid URL.');
    }

    return {
      imageUrl: media.url,
      description: `A generated image of the exoplanet ${input.planet.id}.`,
    };
  }
);

export async function generatePlanetVisual(
  input: GeneratePlanetVisualInput
): Promise<{imageUrl: string; description: string}> {
  return await generatePlanetVisualFlow(input);
}
