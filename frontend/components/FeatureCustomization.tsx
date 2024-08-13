import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

/**
 * FeatureCustomization component
 *
 * This component provides an interface for users to customize features for different omics.
 * It includes dropdowns to select features and sliders to adjust feature values.
 *
 * Props:
 * - features: An array of arrays, each containing strings representing features of different omics.
 * - selectedFeatures: An object mapping each omic index to its selected feature.
 * - featureValues: An object mapping each feature to its current value.
 * - handleFeatureValueChange: A function to handle changes in feature values.
 * - handleSelectChange: A function to handle changes in selected features.
 */
interface FeatureCustomizationProps {
  features: string[][];
  selectedFeatures: { [key: string]: string };
  featureValues: { [key: string]: number };
  handleFeatureValueChange: (feature: string, value: number) => void;
  handleSelectChange: (omicIndex: number, value: string) => void;
}

const FeatureCustomization: React.FC<FeatureCustomizationProps> = ({
  features,
  selectedFeatures,
  featureValues,
  handleFeatureValueChange,
  handleSelectChange,
}) => {
  return (
    <div>
      <div className="text-xl font-semibold tracking-tight py-5">
        Customise Features
      </div>
      <div className="flex flex-col gap-1">
        {features.map((omic, omicIndex) => (
          <div key={omicIndex} className="flex flex-col gap-1">
            <div className="text-xs font-semibold tracking-tight py-2">{`Omic ${
              omicIndex + 1
            }`}</div>
            <Select
              onValueChange={(value) => handleSelectChange(omicIndex, value)}
            >
              <SelectTrigger className="w-full">
                <SelectValue
                  placeholder={`Select feature for Omic ${omicIndex + 1}`}
                />
              </SelectTrigger>
              <SelectContent>
                {omic.map((feature) => (
                  <SelectItem key={feature} value={feature}>
                    {feature}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedFeatures[omicIndex] && (
              <div className="flex items-center gap-2">
                <Slider
                  value={[featureValues[selectedFeatures[omicIndex]] || 0]}
                  max={1}
                  step={0.001}
                  onValueChange={(value) =>
                    handleFeatureValueChange(
                      selectedFeatures[omicIndex],
                      value[0]
                    )
                  }
                  className="py-5"
                />
                <span>
                  {(featureValues[selectedFeatures[omicIndex]] || 0).toFixed(3)}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default FeatureCustomization;
