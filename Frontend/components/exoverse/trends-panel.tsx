
'use client';

import type { Exoplanet } from '@/lib/types';
import { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { DiscoveryOverTimeChart } from './trends/discovery-over-time-chart';
import { ChartContainer } from '../ui/chart';
import { DiscoveryMethodChart } from './trends/discovery-method-chart';
import { PlanetTypeChart } from './trends/planet-type-chart';

const discoveriesChartConfig = {
    Total: { label: 'Total', color: 'hsl(var(--primary))' },
    Kepler: { label: 'Kepler', color: 'hsl(var(--chart-1))' },
    TESS: { label: 'TESS', color: 'hsl(var(--chart-2))' },
    K2: { label: 'K2', color: 'hsl(var(--chart-3))' },
    Other: { label: 'Other', color: 'hsl(var(--chart-4))' },
};

const methodsChartConfig = {
    Transit: { label: 'Transit', color: 'hsl(var(--chart-1))' },
    'Radial Velocity': { label: 'Radial Velocity', color: 'hsl(var(--chart-2))' },
    Imaging: { label: 'Imaging', color: 'hsl(var(--chart-3))' },
    Microlensing: { label: 'Microlensing', color: 'hsl(var(--chart-4))' },
    Other: { label: 'Other', color: 'hsl(var(--chart-5))' },
};

const planetTypesConfig = {
    'Gas Giant': { label: 'Gas Giant', color: 'hsl(var(--chart-1))' },
    'Rocky': { label: 'Rocky', color: 'hsl(var(--chart-2))' },
    'Icy': { label: 'Icy', color: 'hsl(var(--chart-3))' },
    'Extreme': { label: 'Extreme', color: 'hsl(var(--chart-4))' },
};


type TrendsPanelProps = {
    planets: Exoplanet[];
}

export function TrendsPanel({ planets }: TrendsPanelProps) {
    const confirmedPlanets = useMemo(() => planets.filter(p => p.prediction === 'Confirmed'), [planets]);

    const discoveriesByYear = useMemo(() => {
        const byYear: Record<string, any> = {};

        confirmedPlanets.forEach(planet => {
            const year = planet.discoveryYear;
            if (!byYear[year]) {
                byYear[year] = {
                    year: year.toString(), Total: 0, Kepler: 0, TESS: 0, K2: 0, Other: 0,
                };
            }
        });

        const years = Object.keys(byYear).map(Number);
        if (years.length === 0) return [];
        const minYear = Math.min(...years, 2005);
        const maxYear = new Date().getFullYear();

        for(let y = minYear; y <= maxYear; y++) {
            if (!byYear[y]) {
                byYear[y] = { year: y.toString(), Total: 0, Kepler: 0, TESS: 0, K2: 0, Other: 0 };
            }
        }
        
        let cumulative = { Total: 0, Kepler: 0, TESS: 0, K2: 0, Other: 0 };
        const sortedYears = Object.keys(byYear).map(Number).sort((a,b) => a - b);
        
        sortedYears.forEach(year => {
            const yearPlanets = confirmedPlanets.filter(p => p.discoveryYear === year);
            const missionCounts = {
                Kepler: yearPlanets.filter(p => p.mission === 'Kepler').length,
                TESS: yearPlanets.filter(p => p.mission === 'TESS').length,
                K2: yearPlanets.filter(p => p.mission === 'K2').length,
                Other: yearPlanets.filter(p => p.mission !== 'Kepler' && p.mission !== 'TESS' && p.mission !== 'K2').length,
            };
            
            cumulative.Kepler += missionCounts.Kepler;
            cumulative.TESS += missionCounts.TESS;
            cumulative.K2 += missionCounts.K2;
            cumulative.Other += missionCounts.Other;
            cumulative.Total = cumulative.Kepler + cumulative.TESS + cumulative.K2 + cumulative.Other;

            byYear[year] = { ...byYear[year], ...cumulative };
        });

        return Object.values(byYear).sort((a,b) => parseInt(a.year) - parseInt(b.year));
    }, [confirmedPlanets]);

    const methodsByYear = useMemo(() => {
        const byYear: Record<string, any> = {};
        
        const years = Array.from(new Set(confirmedPlanets.map(p => p.discoveryYear))).sort();
        if (years.length === 0) return [];
        const minYear = Math.min(...years, 2005);
        const maxYear = new Date().getFullYear();

        for(let y = minYear; y <= maxYear; y++) {
            byYear[y] = { year: y.toString(), Transit: 0, 'Radial Velocity': 0, Imaging: 0, Microlensing: 0, Other: 0 };
        }

        confirmedPlanets.forEach(planet => {
            if (byYear[planet.discoveryYear]) {
                const method = planet.discoveryMethod;
                if (byYear[planet.discoveryYear][method] !== undefined) {
                    byYear[planet.discoveryYear][method]++;
                } else {
                    byYear[planet.discoveryYear]['Other']++;
                }
            }
        });

        return Object.values(byYear);
    }, [confirmedPlanets]);

    const planetTypeDistribution = useMemo(() => {
        const counts = { 'Gas Giant': 0, 'Rocky': 0, 'Icy': 0, 'Extreme': 0 };
        confirmedPlanets.forEach(p => {
            if (counts[p.planetType] !== undefined) {
                counts[p.planetType]++;
            }
        });
        return Object.entries(counts).map(([type, count]) => ({
            name: type,
            value: count,
            fill: planetTypesConfig[type as keyof typeof planetTypesConfig]?.color || 'hsl(var(--muted))'
        }));
    }, [confirmedPlanets]);
    
    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="glassmorphism lg:col-span-3">
                <CardHeader>
                    <CardTitle>Discoveries Over Time</CardTitle>
                    <CardDescription>
                        Cumulative count of confirmed exoplanet discoveries by mission.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ChartContainer config={discoveriesChartConfig} className="min-h-[400px] w-full">
                        <DiscoveryOverTimeChart data={discoveriesByYear} />
                    </ChartContainer>
                </CardContent>
            </Card>

            <Card className="glassmorphism lg:col-span-2">
                <CardHeader>
                    <CardTitle>Discovery Method Evolution</CardTitle>
                    <CardDescription>
                        Number of confirmed exoplanets discovered by method each year.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ChartContainer config={methodsChartConfig} className="min-h-[300px] w-full">
                        <DiscoveryMethodChart data={methodsByYear} />
                    </ChartContainer>
                </CardContent>
            </Card>

            <Card className="glassmorphism">
                <CardHeader>
                    <CardTitle>Planet Type Distribution</CardTitle>
                    <CardDescription>
                        Distribution of confirmed exoplanet types in the dataset.
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-center">
                    <ChartContainer config={planetTypesConfig} className="min-h-[300px] w-full">
                        <PlanetTypeChart data={planetTypeDistribution} />
                    </ChartContainer>
                </CardContent>
            </Card>
        </div>
    )
}
