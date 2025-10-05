
'use client';

import React, { useState, useMemo } from 'react';
import { Header } from '@/components/layout/header';
import { SkyMap } from '@/components/exoverse/sky-map';
import { Toaster } from '@/components/ui/toaster';
import { EXOPLANETS } from '@/lib/exoplanet-data-large';
import type { Exoplanet } from '@/lib/types';
import { PlanetDetailPanel } from '../exoverse/planet-detail-panel';
import { ApiKeyDialog } from '../ApiKeyDialog';
import { Settings } from 'lucide-react';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CataloguePanel } from '../exoverse/catalogue-panel';
import { ManualEntryPanel } from '../exoverse/manual-entry-panel';
import { TrendsPanel } from '../exoverse/trends-panel';
import { ScrollArea } from '../ui/scroll-area';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion';
import { Checkbox } from '../ui/checkbox';
import { Label } from '../ui/label';

type FilterValues = {
  mission: string[];
  planetType: string[];
  size: string[];
};

type SkyMapFilterValues = {
  Confirmed: boolean;
  Candidate: boolean;
  'False Positive': boolean;
};

type SortValue = 'id' | 'planetRadius' | 'teff' | 'distanceFromStar' | 'confidence';

export default function MainLayout() {
  const [allPlanets, setAllPlanets] = useState<Exoplanet[]>(EXOPLANETS);
  const [selectedPlanet, setSelectedPlanet] = useState<Exoplanet | null>(null);
  const [isDetailPanelOpen, setIsDetailPanelOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // State for Catalogue Filters
  const [catalogueFilters, setCatalogueFilters] = useState<FilterValues>({
    mission: [],
    planetType: [],
    size: [],
  });
  const [catalogueSort, setCatalogueSort] = useState<SortValue>('id');
  const [searchQuery, setSearchQuery] = useState('');
  
  // New, separate state for Sky Map filters
  const [skyMapFilters, setSkyMapFilters] = useState<SkyMapFilterValues>({
    Confirmed: true,
    Candidate: true,
    'False Positive': true,
  });

  const handleSelectPlanet = (planet: Exoplanet | null) => {
    setSelectedPlanet(planet);
    if(planet){
        setIsDetailPanelOpen(true);
    }
  };

  const addPlanet = (planet: Exoplanet) => {
    setAllPlanets(prev => [planet, ...prev]);
  };

  const sortedAndFilteredCataloguePlanets = useMemo(() => {
    let filtered = [...allPlanets];

    if (searchQuery) {
        filtered = filtered.filter(p => p.id.toLowerCase().includes(searchQuery.toLowerCase()));
    }

    if (catalogueFilters.mission.length > 0) {
      filtered = filtered.filter(p => catalogueFilters.mission.includes(p.mission));
    }
    if (catalogueFilters.planetType.length > 0) {
      filtered = filtered.filter(p => catalogueFilters.planetType.includes(p.planetType));
    }
    if (catalogueFilters.size.length > 0) {
        filtered = filtered.filter(p => catalogueFilters.size.includes(p.size));
    }
    
    return filtered.sort((a, b) => {
      if (catalogueSort === 'id') {
        return a.id.localeCompare(b.id);
      }
      const valA = a[catalogueSort] || 0;
      const valB = b[catalogueSort] || 0;
      return valB - valA;
    });
  }, [allPlanets, catalogueFilters, catalogueSort, searchQuery]);
  
  const filteredSkyMapPlanets = useMemo(() => {
      return allPlanets.filter(p => skyMapFilters[p.prediction]);
  }, [allPlanets, skyMapFilters]);
  
  const handleSkyMapFilterChange = (prediction: keyof SkyMapFilterValues, checked: boolean) => {
    setSkyMapFilters(prev => ({...prev, [prediction]: checked }));
  }

  return (
    <div className="flex flex-col h-screen bg-transparent font-body">
      <Header>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsSettingsOpen(true)}
        >
          <Settings />
        </Button>
      </Header>
      <main className="flex-grow pt-16 flex flex-col h-[calc(100vh-4rem)]">
        <Tabs defaultValue="sky-map" className="w-full flex-grow flex flex-col">
          <div className='px-4 pt-4 border-b border-primary/10'>
            <TabsList>
              <TabsTrigger value="sky-map">Sky Map</TabsTrigger>
              <TabsTrigger value="catalogue">Exoplanet Catalogue</TabsTrigger>
              <TabsTrigger value="manual-entry">Manual Entry</TabsTrigger>
              <TabsTrigger value="trends">Trends</TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="sky-map" className="flex-grow relative">
            <SkyMap
              planets={filteredSkyMapPlanets}
              onSelectPlanet={handleSelectPlanet}
              selectedPlanet={selectedPlanet}
            />
            <div className='absolute top-4 left-4 w-full max-w-xs'>
                <Accordion type="single" collapsible defaultValue='filters' className="w-full glassmorphism rounded-lg px-4">
                    <AccordionItem value="filters" className='border-b-0'>
                        <AccordionTrigger className="text-base font-semibold hover:no-underline py-3">Sky Map Filters</AccordionTrigger>
                        <AccordionContent className="space-y-2 pb-2">
                           {(Object.keys(skyMapFilters) as Array<keyof SkyMapFilterValues>).map((key) => (
                             <div key={key} className="flex items-center space-x-2">
                                <Checkbox
                                    id={`sky-filter-${key}`}
                                    checked={skyMapFilters[key]}
                                    onCheckedChange={(checked) => handleSkyMapFilterChange(key, !!checked)}
                                />
                                <Label htmlFor={`sky-filter-${key}`} className='font-normal text-foreground'>{key}</Label>
                             </div>
                           ))}
                        </AccordionContent>
                    </AccordionItem>
                </Accordion>
            </div>
          </TabsContent>

          <TabsContent value="catalogue" className="flex-grow h-0 overflow-hidden">
            <ScrollArea className="h-full">
              <div className="p-4 md:p-6">
                <CataloguePanel 
                  planets={sortedAndFilteredCataloguePlanets}
                  filters={catalogueFilters}
                  setFilters={setCatalogueFilters}
                  sort={catalogueSort}
                  setSort={setCatalogueSort}
                  onSelectPlanet={handleSelectPlanet}
                  searchQuery={searchQuery}
                  setSearchQuery={setSearchQuery}
                />
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="manual-entry" className="flex-grow h-0 overflow-hidden">
            <ScrollArea className="h-full">
              <div className="p-4 md:p-6">
                <ManualEntryPanel addPlanet={addPlanet} />
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="trends" className="flex-grow h-0 overflow-hidden">
            <ScrollArea className="h-full">
              <div className="p-4 md:p-6">
                <TrendsPanel planets={allPlanets} />
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </main>

      <PlanetDetailPanel
        planet={selectedPlanet}
        open={isDetailPanelOpen}
        onOpenChange={setIsDetailPanelOpen}
      />
      <ApiKeyDialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen} />
      <Toaster />
    </div>
  );
}
