"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

/**
 * ProbabilityDistribution component
 *
 * This component renders a bar chart displaying the probability distribution of different classes.
 *
 * Props:
 * - distributions: An array of numbers representing the probability values for different classes.
 *
 * Functionality:
 * - If the distributions array is empty, the component returns null (renders nothing).
 * - Maps the distributions array to a format suitable for the BarChart component.
 * - Renders a bar chart inside a Card component, showing the probability distribution.
 */
const chartConfig = {
  desktop: {
    label: "Probability",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

interface ProbabilityDistributionProps {
  distributions: number[];
}

export function ProbabilityDistribution({
  distributions,
}: ProbabilityDistributionProps) {
  if (distributions.length === 0) {
    return null;
  }

  const chartData = distributions.map((value, index) => ({
    index: `Class ${index}`,
    desktop: value,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Probability Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="index"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
            />
            <YAxis />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar dataKey="desktop" fill="var(--color-desktop)" radius={8} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
