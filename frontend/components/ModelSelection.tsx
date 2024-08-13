import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * ModelSelection component
 *
 * This component provides a dropdown menu for selecting a model from a list of available models.
 *
 * Props:
 * - setSelectedModel: Function to set the selected model.
 * - models: Array of available models for selection.
 *
 * Functionality:
 * - Displays a dropdown menu populated with the list of models.
 * - Calls the setSelectedModel function when a model is selected from the dropdown.
 */
interface ModelSelectionProps {
  setSelectedModel: (model: string) => void;
  models: string[];
}

const ModelSelection: React.FC<ModelSelectionProps> = ({
  setSelectedModel,
  models,
}) => {
  return (
    <Select onValueChange={(value) => setSelectedModel(value)}>
      <SelectTrigger className="w-full">
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model} value={model}>
            {model}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default ModelSelection;
