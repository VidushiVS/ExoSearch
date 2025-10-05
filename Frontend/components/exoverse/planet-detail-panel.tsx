'use client';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { Exoplanet } from '@/lib/types';
import { Telescope, Thermometer, Ruler, Orbit, Star, Scale } from 'lucide-react';
import React, { useState, useEffect } from 'react';
import { generatePlanetVisual } from '@/ai/flows/generate-planet-visuals';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle } from 'lucide-react';
import Image from 'next/image';
import { getApiKey } from '@/ai/client';

type PlanetDetailPanelProps = {
  planet: Exoplanet | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

const InfoRow = ({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) => (
  <div className="flex items-start justify-between py-3 border-b border-primary/10">
    <div className="flex items-center gap-3">
      <div className="text-primary">{icon}</div>
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
    <span className="text-sm font-semibold font-mono text-right">{value}</span>
  </div>
);

const getTemperatureGuidance = (temperature: Exoplanet['temperature']): string => {
    switch (temperature) {
        case 'Hot':
            return 'Visual Emphasis: The planet is HOT. Show signs of intense heat, such as a glowing surface, a shimmering, thick atmosphere, or vast, dry, cracked landscapes. Avoid signs of liquid water or ice.';
        case 'Temperate':
            return 'Visual Emphasis: The planet is TEMPERATE. Give it a conventional, Earth-like appearance with a mix of blue oceans, green landmasses, and white clouds. It should look stable and not exotic.';
        case 'Cold':
            return 'Visual Emphasis: The planet is COLD. It should be dominated by ice and snow, with a frozen-over appearance. The lighting should be crisp and stark, reflecting a frigid environment.';
        default:
            return '';
    }
}

const getDefaultPrompt = (planet: Exoplanet): string => {
  return `Generate a scientifically accurate visualization of the exoplanet ${planet.id}, using real observed data:

Planet radius: ${planet.planetRadius} Earth radii
Stellar temperature: ${planet.teff} K
Equilibrium temperature: ${planet.teff} K
Orbital distance: ${planet.distanceFromStar.toFixed(2)} AU
Star type: G-type main sequence (similar to the Sun)

Render the planet as a scientifically plausible model consistent with its data.
Use correct illumination color, surface conditions, and atmospheric scattering physics.
This image should represent how the planet could realistically appear given these parameters, with no artistic hallucinations or inconsistencies.

${getTemperatureGuidance(planet.temperature)}

Rendering standards:

Use accurate star illumination color based on the spectral type (yellow for G-type).
Apply correct atmospheric scattering.
Model surface materials logically: basaltic rock for hot worlds, silicate sand for arid ones, water ice for cold giants.
Simulate cloud systems, storms, or volcanic activity only if supported by temperature and atmosphere.
Rings, if present, must follow orbital physics.

Lighting and optics:

Use physically realistic exposure, reflection, and albedo values.
Shadows must be accurate to the host star’s angle and intensity.
Avoid exaggerated saturation or unrealistic glow; prioritize NASA-style scientific visualization quality.
Background should be a realistic starfield.

Quality control requirement:

Only finalize and render the image once you are 100% certain the visuals are scientifically consistent, free of visual hallucinations, glitches, or artistic distortions.
If any element appears inconsistent with the planet’s described physics, auto-correct or regenerate silently until a flawless, physically coherent result is achieved.
The final image must look as though captured by a NASA scientific visualization camera or high-fidelity telescope render, not a stylized painting.
The final result should represent a realistic, data-driven world that could exist within a verified exoplanet catalogue.`;
};


export function PlanetDetailPanel({ planet, open, onOpenChange }: PlanetDetailPanelProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (open && planet) {
      const generateImage = async () => {
        setIsLoading(true);
        setError(null);
        setImageUrl(null);
        
        const apiKey = getApiKey();
        if (!apiKey) {
            setError('API key is missing. Please set it in the settings panel.');
            setIsLoading(false);
            return;
        }

        try {
          const prompt = getDefaultPrompt(planet);
          const result = await generatePlanetVisual({ planet, prompt });
          setImageUrl(result.imageUrl);
        } catch (e: any) {
          console.error(e);
          setError(e.message || 'An unexpected error occurred.');
        } finally {
          setIsLoading(false);
        }
      };
      generateImage();
    }
  }, [open, planet]);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-2xl p-0 glassmorphism-sheet">
        <ScrollArea className="h-full">
          {planet && (
            <>
              <div className="w-full h-64 bg-black relative">
                {isLoading && <Skeleton className="w-full h-full" />}
                {error && !isLoading && (
                  <div className="w-full h-full flex items-center justify-center p-4">
                    <Alert variant="destructive" className="max-w-md">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Image Generation Failed</AlertTitle>
                      <AlertDescription>
                        {error}
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
                {imageUrl && !isLoading && (
                  <Image src={imageUrl} alt={`AI rendering of ${planet.id}`} layout="fill" objectFit="cover" />
                )}
                 <div className="absolute bottom-0 left-0 w-full h-24 bg-gradient-to-t from-background to-transparent" />
              </div>

              <SheetHeader className='p-6 -mt-16 relative z-10'>
                <SheetTitle className="text-2xl font-bold font-mono">{planet.id}</SheetTitle>
                
                {imageUrl && !isLoading && !error && (
                  <p className="text-xs text-primary/80 italic pt-1">
                    This visualization represents a scientifically plausible rendering of {planet.id}, generated based on real observational data.
                  </p>
                )}

                <SheetDescription>
                  {planet.details}
                </SheetDescription>
                <div className="flex flex-wrap gap-2 pt-2">
                  <Badge variant="secondary">{planet.mission}</Badge>
                  <Badge variant="secondary">{planet.planetType}</Badge>
                  <Badge variant="secondary">{planet.size}</Badge>
                  <Badge variant="secondary">{planet.temperature}</Badge>
                </div>
              </SheetHeader>
              <div className="px-6 pb-6 space-y-6">
                <div>
                  <h4 className="font-semibold text-lg mb-2 text-primary">Planet Properties</h4>
                  <InfoRow icon={<Ruler size={18} />} label="Planet Radius" value={`${planet.planetRadius} R⊕`} />
                  <InfoRow icon={<Thermometer size={18} />} label="Effective Temp." value={`${planet.teff} K`} />
                  <InfoRow icon={<Orbit size={18} />} label="Orbital Period" value={`${planet.orbitalPeriod} days`} />
                  <InfoRow icon={<Scale size={18} />} label="a/R*" value={planet.a_R} />
                  <InfoRow icon={<Ruler size={18} />} label="SNR" value={planet.snr} />
                </div>
                
                <div>
                  <h4 className="font-semibold text-lg mb-2 text-primary">Stellar Properties</h4>
                  <InfoRow icon={<Star size={18} />} label="Stellar Radius" value={`${planet.stellarRadius} R☉`} />
                  <InfoRow icon={<Telescope size={18} />} label="Magnitude" value={planet.magnitude} />
                  <InfoRow icon={<Thermometer size={18} />} label="log(g)" value={planet.logg} />
                </div>

                <div>
                  <h4 className="font-semibold text-lg mb-2 text-primary">Vetting Metrics</h4>
                  <InfoRow icon={<Scale size={18} />} label="Odd/Even Ratio" value={planet.oddEvenRatio} />
                  <InfoRow icon={<Scale size={18} />} label="Crowding" value={planet.crowding} />
                  <InfoRow icon={<Scale size={18} />} label="Contamination" value={planet.contamination} />
                </div>
              </div>
            </>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
