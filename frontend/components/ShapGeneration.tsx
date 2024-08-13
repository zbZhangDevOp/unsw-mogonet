import React from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ShapGenerationProps {
  shapUrl: string | null;
  generatingShap: boolean;
  handleShap: () => void;
}

/**
 * ShapGeneration component
 *
 * This component is responsible for generating and displaying SHAP (SHapley Additive exPlanations) visualizations.
 * It provides a button to trigger SHAP generation and a dialog to display the generated SHAP visualization.
 *
 * Props:
 * - shapUrl: URL of the generated SHAP visualization.
 * - generatingShap: Boolean indicating whether the SHAP generation process is ongoing.
 * - handleShap: Function to handle the SHAP generation process.
 *
 * Functionality:
 * - Displays a "Generate Shap" button to initiate SHAP generation.
 * - Shows a "Show Shap" button if SHAP URL is available, opening a dialog with the SHAP image.
 */
const ShapGeneration: React.FC<ShapGenerationProps> = ({
  shapUrl,
  generatingShap,
  handleShap,
}) => {
  return (
    <div className="flex justify-start gap-2">
      <Button onClick={handleShap} className="mt-10" disabled={generatingShap}>
        {generatingShap ? "Generating..." : "Generate Shap"}
      </Button>
      <Dialog>
        <DialogTrigger>
          {shapUrl && (
            <Button
              className="mt-10"
              variant={"outline"}
              disabled={generatingShap}
            >
              Show Shap
            </Button>
          )}
        </DialogTrigger>
        <DialogContent className="h-3/4 w-3/4">
          <DialogHeader className="h-full w-full flex justify-center items-center">
            <img
              src={shapUrl || ""}
              alt="Shap"
              width={500}
              height={500}
              className="object-contain h-full w-2/3"
            />
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ShapGeneration;
