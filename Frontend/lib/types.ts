
export type Exoplanet = {
  id: string; // Object ID
  mission: 'Kepler' | 'TESS' | 'K2' | 'Manual' | 'Other';
  prediction: 'Confirmed' | 'Candidate' | 'False Positive';
  confidence: number;
  details: string;
  ra: number; // 0-360
  dec: number; // -90 to 90
  magnitude: number;
  orbitalPeriod: number;
  duration: number;
  depth: number;
  snr: number;
  a_R: number; // a/R*
  planetRadius: number; // Earth radii
  teff: number; // Kelvin
  logg: number;
  stellarRadius: number; // Solar radii
  oddEvenRatio: number;
  secondaryDepth: number;
  crowding: number;
  contamination: number;
  discoveryYear: number;
  discoveryMethod: 'Transit' | 'Radial Velocity' | 'Imaging' | 'Microlensing' | 'Other';
  // Derived/simplified properties for UI
  planetType: 'Gas Giant' | 'Rocky' | 'Icy' | 'Extreme';
  size: 'Small' | 'Medium' | 'Large' | 'Super-Jupiter'; // Derived from planetRadius
  temperature: 'Hot' | 'Temperate' | 'Cold'; // Derived from teff and a_R
  orbitalDetails: string;
  distanceFromStar: number; // in AU
};
