"use client";

import React from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Component, Package, Triangle, Upload } from "lucide-react";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import Link from "next/link";

/**
 * SideNav component
 *
 * This component renders a side navigation bar with links to different pages of the application.
 * It highlights the active link based on the current pathname and provides tooltips for each navigation item.
 *
 * Functionality:
 * - Displays navigation buttons with icons and tooltips.
 * - Highlights the active navigation button based on the current pathname.
 *
 * Routes:
 * - Upload Dataset: Navigates to the dataset upload page.
 * - Manage Datasets: Navigates to the dataset management page.
 * - Train and Test: Navigates to the training and testing page.
 */
export default function SideNav() {
  const pathname = usePathname();

  const routes = [
    {
      href: `/upload`,
      label: "Upload Dataset",
      icon: <Upload className="size-5" />,
      active: pathname === `/upload`,
    },
    {
      href: `/`,
      label: "Manage Datasets",
      icon: <Component className="size-5" />,
      active: pathname === `/`,
    },
    {
      href: `/training`,
      label: "Train and Test",
      icon: <Package className="size-5" />,
      active: pathname === `/training`,
    },
  ];

  return (
    <aside className="inset-y fixed  left-0 z-20 flex h-full flex-col border-r">
      <div className="border-b p-2">
        <Button variant="outline" size="icon" aria-label="Home">
          <Triangle className="size-5 fill-foreground" />
        </Button>
      </div>
      <nav className="grid gap-1 p-2">
        {routes.map((route) => (
          <Tooltip key={route.href}>
            <TooltipTrigger asChild>
              <Link href={route.href} className="w-full">
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(route.active && "bg-muted", "rounded-lg")}
                  aria-label={route.label}
                >
                  {route.icon}
                </Button>
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right" sideOffset={5}>
              {route.label}
            </TooltipContent>
          </Tooltip>
        ))}
      </nav>
    </aside>
  );
}
