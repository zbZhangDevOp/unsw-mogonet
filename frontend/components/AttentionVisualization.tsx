import React from "react";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTrigger,
} from "@/components/ui/dialog";

/**
 * Props interface for AttentionVisualization component.
 * @property {string | null} graphUrl - URL for the graph image.
 * @property {string | null} heatmapUrl - URL for the heatmap image.
 */
interface AttentionVisualizationProps {
  graphUrl: string | null;
  heatmapUrl: string | null;
}

/**
 * AttentionVisualization component is responsible for displaying attention visualizations.
 * It shows two types of visualizations: a graph and a heatmap.
 * Each visualization is displayed within a dialog (modal) that can be triggered by clicking on the image.
 *
 * @param {AttentionVisualizationProps} props - Props containing URLs for the graph and heatmap images.
 * @returns {JSX.Element} The rendered AttentionVisualization component.
 */
const AttentionVisualization: React.FC<AttentionVisualizationProps> = ({
  graphUrl,
  heatmapUrl,
}) => {
  return (
    <div>
      <div className="text-2xl font-semibold tracking-tight">Attention</div>
      <div className="flex justify-between space-x-4 mt-20">
        <Dialog>
          <DialogTrigger>
            {graphUrl && (
              <div className="flex-grow flex flex-col justify-center items-center">
                <Label>Graph</Label>
                <img
                  src={graphUrl}
                  alt="Graph"
                  width={500}
                  height={500}
                  className="object-contain"
                />
              </div>
            )}
          </DialogTrigger>
          <DialogContent className="h-3/4 w-3/4">
            <DialogHeader className="h-full w-full flex justify-center items-center">
              <img
                src={graphUrl || ""}
                alt="Graph"
                width={500}
                height={500}
                className="object-contain h-full w-2/3"
              />
            </DialogHeader>
          </DialogContent>
        </Dialog>
        <Dialog>
          <DialogTrigger>
            {heatmapUrl && (
              <div className="flex-grow flex flex-col justify-center items-center">
                <Label>Heatmap</Label>
                <img
                  src={heatmapUrl}
                  alt="Heatmap"
                  width={500}
                  height={500}
                  className="object-contain"
                />
              </div>
            )}
          </DialogTrigger>
          <DialogContent className="h-3/4 w-3/4">
            <DialogHeader className="h-full w-full flex justify-center items-center">
              <img
                src={heatmapUrl || ""}
                alt="Heatmap"
                width={500}
                height={500}
                className="object-contain h-full w-2/3"
              />
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
};

export default AttentionVisualization;
