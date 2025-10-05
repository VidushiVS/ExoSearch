import {z} from 'genkit';

// Define the input schema for the planet visual generation.
export const GeneratePlanetVisualInputSchema = z.object({
  planet: z.any(),
  prompt: z.string(),
});
export type GeneratePlanetVisualInput = z.infer<
  typeof GeneratePlanetVisualInputSchema
>;
