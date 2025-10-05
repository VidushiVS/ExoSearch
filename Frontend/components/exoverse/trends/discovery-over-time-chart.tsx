
'use client';
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

const chartConfig = {
    Total: { label: 'Total', color: 'hsl(var(--primary))' },
    Kepler: { label: 'Kepler', color: 'hsl(var(--chart-1))' },
    TESS: { label: 'TESS', color: 'hsl(var(--chart-2))' },
    K2: { label: 'K2', color: 'hsl(var(--chart-3))' },
    Other: { label: 'Other', color: 'hsl(var(--chart-4))' },
  } satisfies ChartConfig;

type DiscoveryOverTimeChartProps = {
    data: any[];
}

export function DiscoveryOverTimeChart({ data }: DiscoveryOverTimeChartProps) {
    return (
        <ResponsiveContainer width="100%" height="100%">
            <AreaChart
                data={data}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" vertical={false} />
                <XAxis 
                    dataKey="year" 
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    tickFormatter={(value) => value.toString()}
                />
                <YAxis 
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    tickFormatter={(value) => value.toLocaleString()}
                />
                <Tooltip 
                    cursor={{ stroke: 'hsl(var(--primary))', strokeWidth: 1, strokeDasharray: '3 3' }}
                    content={<ChartTooltipContent indicator="line" />} 
                />
                <Legend content={<ChartLegendContent />} />

                {Object.keys(chartConfig).map((mission) => (
                    <defs key={mission}>
                        <linearGradient id={`color${mission}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={chartConfig[mission as keyof typeof chartConfig].color} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={chartConfig[mission as keyof typeof chartConfig].color} stopOpacity={0.1} />
                        </linearGradient>
                    </defs>
                ))}
                
                {(Object.keys(chartConfig) as Array<keyof typeof chartConfig>).map((mission) => (
                    <Area
                        key={mission}
                        type="monotone"
                        dataKey={mission}
                        stackId={mission === 'Total' ? undefined : "1"}
                        stroke={chartConfig[mission].color}
                        fillOpacity={1}
                        fill={`url(#color${mission})`}
                        strokeWidth={2}
                        dot={false}
                    />
                ))}
            </AreaChart>
        </ResponsiveContainer>
    );
}
