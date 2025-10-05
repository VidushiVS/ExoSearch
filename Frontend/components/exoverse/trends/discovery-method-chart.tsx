
'use client';
import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar, BarChart } from "recharts";
import { ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

const chartConfig = {
    Transit: { label: 'Transit', color: 'hsl(var(--chart-1))' },
    'Radial Velocity': { label: 'Radial Velocity', color: 'hsl(var(--chart-2))' },
    Imaging: { label: 'Imaging', color: 'hsl(var(--chart-3))' },
    Microlensing: { label: 'Microlensing', color: 'hsl(var(--chart-4))' },
    Other: { label: 'Other', color: 'hsl(var(--chart-5))' },
} satisfies ChartConfig;

type DiscoveryMethodChartProps = {
    data: any[];
}

export function DiscoveryMethodChart({ data }: DiscoveryMethodChartProps) {
    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart
                data={data}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                stackOffset="expand" // This creates a 100% stacked bar chart
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
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip 
                    cursor={{ fill: 'hsl(var(--primary) / 0.1)' }}
                    content={<ChartTooltipContent indicator="dot" />}
                />
                <Legend content={<ChartLegendContent />} />

                {(Object.keys(chartConfig) as Array<keyof typeof chartConfig>).map((method) => (
                    <Bar
                        key={method}
                        dataKey={method}
                        stackId="a"
                        fill={chartConfig[method].color}
                        radius={[2, 2, 0, 0]}
                    />
                ))}
            </BarChart>
        </ResponsiveContainer>
    );
}
