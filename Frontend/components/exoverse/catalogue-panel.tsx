'use client';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
  } from "@/components/ui/accordion"
import type { Exoplanet } from '@/lib/types';
import { ScrollArea } from '../ui/scroll-area';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Eye, FileDown, Search } from 'lucide-react';
import { exportToCSV } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { Input } from '../ui/input';

type FilterValues = {
  mission: string[];
  planetType: string[];
  size: string[];
};

type SortValue = 'id' | 'planetRadius' | 'teff' | 'distanceFromStar' | 'confidence';

type CataloguePanelProps = {
  planets: Exoplanet[];
  filters: FilterValues;
  setFilters: React.Dispatch<React.SetStateAction<FilterValues>>;
  sort: SortValue;
  setSort: React.Dispatch<React.SetStateAction<SortValue>>;
  onSelectPlanet: (planet: Exoplanet) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
};

const missionOptions = ['Kepler', 'TESS', 'K2', 'Manual'];
const planetTypeOptions = ['Gas Giant', 'Rocky', 'Icy', 'Extreme'];
const sizeOptions = ['Small', 'Medium', 'Large', 'Super-Jupiter'];

const predictionStyles: Record<Exoplanet['prediction'], { badge: string, progress: string }> = {
    'Confirmed': { badge: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30', progress: '[&>div]:bg-cyan-500' },
    'Candidate': { badge: 'bg-orange-500/20 text-orange-400 border-orange-500/30 hover:bg-orange-500/30', progress: '[&>div]:bg-orange-500' },
    'False Positive': { badge: 'bg-red-500/20 text-red-400 border-red-500/30 hover:bg-red-500/30', progress: '[&>div]:bg-red-500' }
}

export function CataloguePanel({
  planets,
  filters,
  setFilters,
  sort,
  setSort,
  onSelectPlanet,
  searchQuery,
  setSearchQuery,
}: CataloguePanelProps) {
  const { toast } = useToast();

  const handleCheckboxChange = (category: keyof FilterValues, value: string, checked: boolean) => {
    setFilters(prev => {
        const newValues = checked ? [...prev[category], value] : prev[category].filter(v => v !== value);
        return { ...prev, [category]: newValues };
    });
  }

  const handleExport = () => {
    exportToCSV(planets, 'exoplanet_catalogue');
    toast({
        title: 'Export Successful',
        description: `${planets.length} records exported to exoplanet_catalogue.csv`,
    });
  }

  return (
    <div className="space-y-6">
        <Accordion type="single" collapsible defaultValue='filters' className="w-full glassmorphism rounded-lg px-6">
            <AccordionItem value="filters" className='border-b-0'>
                <AccordionTrigger className="text-lg font-semibold hover:no-underline">Filters, Sorting & Export</AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                         <div className='col-span-1 md:col-span-3'>
                             <Label htmlFor="search-planet" className="text-sm font-medium text-muted-foreground">Search by Name</Label>
                             <div className="relative mt-3">
                                 <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                                 <Input 
                                     id="search-planet"
                                     placeholder="e.g., Kepler-186f"
                                     className="pl-10"
                                     value={searchQuery}
                                     onChange={(e) => setSearchQuery(e.target.value)}
                                 />
                             </div>
                         </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-5 gap-6 pt-4">
                        <div>
                            <Label className="text-sm font-medium text-muted-foreground">Mission</Label>
                            <div className="flex flex-col space-y-2 mt-3">
                            {missionOptions.map(option => (
                                <div key={option} className="flex items-center space-x-2">
                                <Checkbox id={`mission-${option}`} checked={filters.mission.includes(option)} onCheckedChange={(checked) => handleCheckboxChange('mission', option, !!checked)} />
                                <Label htmlFor={`mission-${option}`} className='font-normal text-foreground'>{option}</Label>
                                </div>
                            ))}
                            </div>
                        </div>
                        <div>
                            <Label className="text-sm font-medium text-muted-foreground">Planet Type</Label>
                            <div className="flex flex-col space-y-2 mt-3">
                            {planetTypeOptions.map(option => (
                                <div key={option} className="flex items-center space-x-2">
                                <Checkbox id={`type-${option}`} checked={filters.planetType.includes(option)} onCheckedChange={(checked) => handleCheckboxChange('planetType', option, !!checked)} />
                                <Label htmlFor={`type-${option}`} className='font-normal text-foreground'>{option}</Label>
                                </div>
                            ))}
                            </div>
                        </div>
                        <div>
                            <Label className="text-sm font-medium text-muted-foreground">Size</Label>
                            <div className="flex flex-col space-y-2 mt-3">
                            {sizeOptions.map(option => (
                                <div key={option} className="flex items-center space-x-2">
                                <Checkbox id={`size-${option}`} checked={filters.size.includes(option)} onCheckedChange={(checked) => handleCheckboxChange('size', option, !!checked)} />
                                <Label htmlFor={`size-${option}`} className='font-normal text-foreground'>{option}</Label>
                                </div>
                            ))}
                            </div>
                        </div>
                        <div>
                            <Label htmlFor="sort-by" className="text-sm font-medium text-muted-foreground">Sort By</Label>
                            <Select value={sort} onValueChange={(value: SortValue) => setSort(value)}>
                                <SelectTrigger id="sort-by" className="w-full mt-3">
                                <SelectValue placeholder="Sort by..." />
                                </SelectTrigger>
                                <SelectContent>
                                <SelectItem value="id">Name</SelectItem>
                                <SelectItem value="confidence">Confidence</SelectItem>
                                <SelectItem value="planetRadius">Planet Radius</SelectItem>
                                <SelectItem value="teff">Temperature</SelectItem>
                                <SelectItem value="distanceFromStar">Distance from Star</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                         <div>
                            <Label className="text-sm font-medium text-muted-foreground">Export Data</Label>
                            <Button onClick={handleExport} className='w-full mt-3'>
                                <FileDown className="mr-2 h-4 w-4" />
                                Export as CSV
                            </Button>
                        </div>
                    </div>
                </AccordionContent>
            </AccordionItem>
        </Accordion>
        
        <div className="space-y-3">
            <h3 className="text-xl font-semibold">Exoplanet Catalogue ({planets.length} results)</h3>
            <div className="border rounded-lg glassmorphism">
                <ScrollArea className="h-[65vh]">
                <Table>
                    <TableHeader className='sticky top-0 z-10 bg-background/50 backdrop-blur-md'>
                        <TableRow>
                            <TableHead className='w-[200px]'>Object ID</TableHead>
                            <TableHead>Mission</TableHead>
                            <TableHead>Prediction</TableHead>
                            <TableHead className='w-[150px]'>Confidence</TableHead>
                            <TableHead>Radius (R⊕)</TableHead>
                            <TableHead>Temp (K)</TableHead>
                            <TableHead className='text-right'>Actions</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                    {planets.map(planet => (
                        <TableRow key={planet.id} className="group transition-colors hover:bg-primary/10">
                            <TableCell className="font-medium font-mono">{planet.id}</TableCell>
                            <TableCell>{planet.mission}</TableCell>
                            <TableCell><Badge className={predictionStyles[planet.prediction].badge}>{planet.prediction}</Badge></TableCell>
                            <TableCell>
                                <div className='flex items-center gap-2'>
                                    <Progress value={planet.confidence * 100} className={`h-2 ${predictionStyles[planet.prediction].progress}`} />
                                    <span className='text-xs font-mono text-muted-foreground'>{ (planet.confidence * 100).toFixed(0) }%</span>
                                </div>
                            </TableCell>
                            <TableCell className="font-mono">{planet.planetRadius.toFixed(2)}</TableCell>
                            <TableCell className="font-mono">{planet.teff}</TableCell>
                            <TableCell className="text-right">
                                <Button variant="ghost" size="icon" onClick={() => onSelectPlanet(planet)} className='opacity-50 group-hover:opacity-100 transition-opacity hover:scale-105 hover:text-primary'>
                                    <Eye className='h-4 w-4' />
                                </Button>
                            </TableCell>
                        </TableRow>
                    ))}
                    </TableBody>
                </Table>
                </ScrollArea>
            </div>
        </div>
    </div>
  );
}
