'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import type { Exoplanet } from '@/lib/types';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

const formSchema = z.object({
  mission: z.enum(['Kepler', 'TESS', 'K2']),
  objectName: z.string().min(1, 'Object Name is required'),
  ra: z.coerce.number().min(0).max(360),
  dec: z.coerce.number().min(-90).max(90),
  magnitude: z.coerce.number(),
  orbitalPeriod: z.coerce.number().positive(),
  duration: z.coerce.number().positive(),
  depth: z.coerce.number().positive(),
  snr: z.coerce.number(),
  a_R: z.coerce.number(),
  planetRadius: z.coerce.number().positive(),
  teff: z.coerce.number().positive(),
  logg: z.coerce.number(),
  stellarRadius: z.coerce.number().positive(),
  oddEvenRatio: z.coerce.number(),
  secondaryDepth: z.coerce.number(),
  crowding: z.coerce.number(),
  contamination: z.coerce.number(),
});

type ManualEntryPanelProps = {
  addPlanet: (planet: Exoplanet) => void;
}

export function ManualEntryPanel({ addPlanet }: ManualEntryPanelProps) {
  const { toast } = useToast();
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
        objectName: 'MyPlanet-01',
        mission: 'Kepler',
        ra: 180,
        dec: 0,
        magnitude: 15,
        orbitalPeriod: 365,
        duration: 8,
        depth: 1000,
        snr: 50,
        a_R: 20,
        planetRadius: 1,
        teff: 5700,
        logg: 4.5,
        stellarRadius: 1,
        oddEvenRatio: 0.1,
        secondaryDepth: 0,
        crowding: 1,
        contamination: 0
    }
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    const newPlanet: Exoplanet = {
      id: values.objectName,
      mission: 'Manual',
      prediction: 'Candidate',
      confidence: 1.0,
      details: 'Manually entered exoplanet.',
      ra: values.ra,
      dec: values.dec,
      magnitude: values.magnitude,
      orbitalPeriod: values.orbitalPeriod,
      duration: values.duration,
      depth: values.depth,
      snr: values.snr,
      a_R: values.a_R,
      planetRadius: values.planetRadius,
      teff: values.teff,
      logg: values.logg,
      stellarRadius: values.stellarRadius,
      oddEvenRatio: values.oddEvenRatio,
      secondaryDepth: values.secondaryDepth,
      crowding: values.crowding,
      contamination: values.contamination,
      planetType: 'Rocky', // Default values for derived properties
      size: 'Medium',
      temperature: 'Temperate',
      orbitalDetails: 'Manually specified orbit.',
      distanceFromStar: Math.sqrt(values.orbitalPeriod / 365.25),
    };

    addPlanet(newPlanet);

    toast({
        title: "Prediction Submitted",
        description: `${values.objectName} has been added to the sky map.`,
    });
    form.reset();
  }

  return (
    <Card className="border-0 shadow-none bg-transparent">
      <CardHeader>
        <CardTitle className="text-xl font-semibold">Enter Exoplanet Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <div className='p-6 rounded-lg glassmorphism'>
              <p className='text-base font-semibold text-primary mb-4'>Basic Information</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField
                  control={form.control}
                  name="mission"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Mission</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger><SelectValue placeholder="Select mission" /></SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="Kepler">Kepler</SelectItem>
                          <SelectItem value="TESS">TESS</SelectItem>
                          <SelectItem value="K2">K2</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField control={form.control} name="objectName" render={({ field }) => (
                    <FormItem><FormLabel>Object Name</FormLabel><FormControl><Input placeholder="e.g., KOI-123.01" {...field} /></FormControl><FormMessage /></FormItem>
                )} />
                <FormField control={form.control} name="magnitude" render={({ field }) => (
                    <FormItem><FormLabel>Magnitude</FormLabel><FormControl><Input type="number" placeholder="e.g., 12.5" {...field} /></FormControl><FormMessage /></FormItem>
                )} />
                <FormField control={form.control} name="ra" render={({ field }) => (
                    <FormItem><FormLabel>Right Ascension (RA)</FormLabel><FormControl><Input type="number" placeholder="0-360 degrees" {...field} /></FormControl><FormMessage /></FormItem>
                )} />
                <FormField control={form.control} name="dec" render={({ field }) => (
                    <FormItem><FormLabel>Declination (Dec)</FormLabel><FormControl><Input type="number" placeholder="-90 to 90 degrees" {...field} /></FormControl><FormMessage /></FormItem>
                )} />
              </div>
            </div>
            
            <div className='p-6 rounded-lg glassmorphism'>
              <p className='text-base font-semibold text-primary mb-4'>Transit Parameters</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <FormField control={form.control} name="orbitalPeriod" render={({ field }) => (
                      <FormItem><FormLabel>Orbital Period (days)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="duration" render={({ field }) => (
                      <FormItem><FormLabel>Duration (hours)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="depth" render={({ field }) => (
                      <FormItem><FormLabel>Depth (ppm)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="snr" render={({ field }) => (
                      <FormItem><FormLabel>Signal-to-Noise (SNR)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="a_R" render={({ field }) => (
                      <FormItem><FormLabel>a/R*</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
              </div>
            </div>

            <div className='p-6 rounded-lg glassmorphism'>
              <p className='text-base font-semibold text-primary mb-4'>Stellar Parameters</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <FormField control={form.control} name="planetRadius" render={({ field }) => (
                      <FormItem><FormLabel>Planet Radius (R⊕)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="teff" render={({ field }) => (
                      <FormItem><FormLabel>Effective Temp (K)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="logg" render={({ field }) => (
                      <FormItem><FormLabel>log(g)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="stellarRadius" render={({ field }) => (
                      <FormItem><FormLabel>Stellar Radius (R☉)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
              </div>
            </div>

            <div className='p-6 rounded-lg glassmorphism'>
              <p className='text-base font-semibold text-primary mb-4'>Vetting Metrics</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <FormField control={form.control} name="oddEvenRatio" render={({ field }) => (
                      <FormItem><FormLabel>Odd/Even Ratio</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="secondaryDepth" render={({ field }) => (
                      <FormItem><FormLabel>Secondary Depth (ppm)</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="crowding" render={({ field }) => (
                      <FormItem><FormLabel>Crowding</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
                  <FormField control={form.control} name="contamination" render={({ field }) => (
                      <FormItem><FormLabel>Contamination</FormLabel><FormControl><Input type="number" {...field} /></FormControl><FormMessage /></FormItem>
                  )} />
              </div>
            </div>

            <Button type="submit" size="lg" className="w-full md:w-auto transition-transform hover:scale-105">Submit for Prediction</Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
