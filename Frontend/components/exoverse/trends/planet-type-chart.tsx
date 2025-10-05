
'use client';
import { Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

type PlanetTypeChartProps = {
    data: any[];
}

export function PlanetTypeChart({ data }: PlanetTypeChartProps) {
    return (
        <ResponsiveContainer width="100%" height="100%">
            <PieChart>
                <Tooltip content={<ChartTooltipContent />} />
                <Pie 
                    data={data} 
                    dataKey="value" 
                    nameKey="name"
                    cx="50%" 
                    cy="50%" 
                    outerRadius="80%"
                    innerRadius="50%"
                    labelLine={false}
                    label={({
                      cx,
                      cy,
                      midAngle,
                      innerRadius,
                      outerRadius,
                      value,
                      index,
                    }) => {
                      const RADIAN = Math.PI / 180
                      const radius = 12 + innerRadius + (outerRadius - innerRadius)
                      const x = cx + radius * Math.cos(-midAngle * RADIAN)
                      const y = cy + radius * Math.sin(-midAngle * RADIAN)
         
                      return (
                        <text
                          x={x}
                          y={y}
                          className="fill-muted-foreground text-xs"
                          textAnchor={x > cx ? "start" : "end"}
                          dominantBaseline="central"
                        >
                          {data[index].name} ({value})
                        </text>
                      )
                    }}
                />
            </PieChart>
        </ResponsiveContainer>
    );
}
