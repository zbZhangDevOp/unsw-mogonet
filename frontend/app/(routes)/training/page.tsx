"use client";

import React, { use, useEffect, useState } from "react";
import axios from "axios";
import { useForm, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import HyperparametersForm from "@/components/HyperparametersForm";
import FeatureCustomization from "@/components/FeatureCustomization";
import ShapGeneration from "@/components/ShapGeneration";
import AttentionVisualization from "@/components/AttentionVisualization";
import { ProbabilityDistribution } from "@/components/ProbabilityDistribution";
import { z } from "zod";
import toast from "react-hot-toast";

export const BACKEND_BASE_URL = "http://localhost:8000";

interface FeatureValues {
  [key: string]: any; // Adjust the type as needed
}

const formSchema = z.object({
  model_name: z.string().nonempty("Model name is required."),
  hyperparameters: z
    .string()
    .nonempty("Hyperparameter must be a positive number."),
});

type FormValues = z.infer<typeof formSchema>;

export default function TrainingPage() {
  const [features, setFeatures] = useState<string[][]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<{
    [key: string]: string;
  }>({});
  const [featureValues, setFeatureValues] = useState<{ [key: string]: number }>(
    {}
  );
  const [sampleData, setSampleData] = useState<number[]>([]);
  const [presetFeatureValues, setPresetFeatureValues] = useState<
    FeatureValues[]
  >([]);
  const [distributions, setDistributions] = useState<number[]>([]);
  const [graphUrl, setGraphUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [shapUrl, setShapUrl] = useState<string | null>(null);
  const [generatingShap, setGeneratingShap] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get(`${BACKEND_BASE_URL}/get-all-dataset`);

        setModels(response.data.datasets);
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    fetchModels();
  }, []);

  const formMethods = useForm<FormValues>({
    resolver: zodResolver(formSchema),
  });

  const handleFeatureValueChange = (feature: string, value: number) => {
    setFeatureValues((prevValues) => ({
      ...prevValues,
      [feature]: value,
    }));
  };

  const handleSelectChange = (omicIndex: number, value: string) => {
    setSelectedFeatures((prevSelected) => ({
      ...prevSelected,
      [omicIndex]: value,
    }));
  };

  const setDefaultFeatureValues = (value: string) => {
    let combinedDictionary = {};
    for (let x = 0; x < presetFeatureValues.length; x++) {
      combinedDictionary = {
        ...combinedDictionary,
        ...presetFeatureValues[x][Number(value) - 1],
      };
    }
    setFeatureValues(combinedDictionary);
  };

  const handleSubmitFeatureValues = async () => {
    try {
      const featureValuesList: number[][] = features.map((sublist) =>
        sublist.map((feature) => featureValues[feature] || 0)
      );

      const modelName = selectedModel;

      const formData = new FormData();
      formData.append("features_list", JSON.stringify(featureValuesList));

      const response = await axios.post(
        `${BACKEND_BASE_URL}/probability-distribution/${modelName}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setDistributions(response.data.distributions[0]);
    } catch (error) {
      console.error("Error submitting feature values:", error);
    }
  };

  const handleShap = async () => {
    try {
      setGeneratingShap(true);
      const featureValuesList: number[][] = features.map((sublist) =>
        sublist.map((feature) => featureValues[feature] || 0)
      );

      const modelName = selectedModel;

      const formData = new FormData();
      formData.append("features_list", JSON.stringify(featureValuesList));

      const response = await axios.post(
        `${BACKEND_BASE_URL}/shap_values/${modelName}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const timestamp = new Date().getTime();
      setShapUrl(
        `${BACKEND_BASE_URL}/${response.data.shap_values.path}?t=${timestamp}`
      );
      setGeneratingShap(false);
    } catch (error) {
      console.error("Error submitting feature values:", error);
    }
  };

  const onSubmit = async (values: FormValues) => {
    try {
      setIsTraining(true);
      const formData = new FormData();
      formData.append("adj_parameter", values.hyperparameters.toString());

      const response = await axios.post(
        `${BACKEND_BASE_URL}/train-model/${values.model_name}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const featureResponse = await axios.get(
        `${BACKEND_BASE_URL}/dataset/data/${values.model_name}`
      );

      const parsedFeature = JSON.parse(featureResponse.data);

      const featureValues = parsedFeature.features;

      const featureValueLength = featureValues[0].length;

      setPresetFeatureValues(featureValues);

      setSampleData([...Array(featureValueLength)].map((_, i) => i + 1));

      const fetchedFeatures: string[][] = parsedFeature.feature_names;

      setFeatures(fetchedFeatures);

      // Initialize feature values dictionary
      const initialFeatureValues: { [key: string]: number } = {};
      fetchedFeatures.flat().forEach((feature) => {
        initialFeatureValues[feature] = 0;
      });

      setFeatureValues(initialFeatureValues);

      // Fetch the attention visualization images
      const attentionResponse = await axios.post(
        `${BACKEND_BASE_URL}/attention-visualisation/${values.model_name}`
      );

      setGraphUrl(`${BACKEND_BASE_URL}/${attentionResponse.data.graph.path}`);
      setHeatmapUrl(
        `${BACKEND_BASE_URL}/${attentionResponse.data.heatmap.path}`
      );
      setIsTraining(false);
    } catch (error: any) {
      setIsTraining(false);
      if (error.response) {
        // Server responded with a status other than 200 range
        console.error("Error:", error.response.data.detail); // Accessing the custom detail message
        toast.error(`Error: ${error.response.data.detail}`);
      } else if (error.request) {
        // Request was made but no response received
        console.error("No response received:", error.request);
      } else {
        // Something else happened
        console.error("Error:", error.message);
      }
    }
  };

  useEffect(() => {
    handleSubmitFeatureValues();
  }, [featureValues]);

  return (
    <FormProvider {...formMethods}>
      <div className="grid grid-rows-3 col-span-2 p-5 gap-5 w-full">
        <div>
          <div className="text-2xl font-semibold tracking-tight mb-10">
            Model Training
          </div>
          <HyperparametersForm
            onSubmit={onSubmit}
            isTraining={isTraining}
            setSelectedModel={setSelectedModel}
            models={models}
          />
        </div>
        <div className="row-span-2">
          <div className="text-2xl font-semibold tracking-tight mb-10">
            Features
          </div>
          {features.length > 0 && (
            <>
              <Label className="pt-10 mb-5">
                Select initial sample as feature values
              </Label>
              <Select onValueChange={(value) => setDefaultFeatureValues(value)}>
                <SelectTrigger className="w-full mt-2">
                  <SelectValue
                    placeholder={`Select initial sample as feature values`}
                  />
                </SelectTrigger>
                <SelectContent>
                  {sampleData.map((sample) => (
                    <SelectItem key={sample} value={sample.toString()}>
                      {`Sample ${sample}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          )}
          {features.length > 0 && (
            <FeatureCustomization
              features={features}
              selectedFeatures={selectedFeatures}
              featureValues={featureValues}
              handleFeatureValueChange={handleFeatureValueChange}
              handleSelectChange={handleSelectChange}
            />
          )}
          {features.length > 0 && (
            <ShapGeneration
              shapUrl={shapUrl}
              generatingShap={generatingShap}
              handleShap={handleShap}
            />
          )}
        </div>
      </div>
      <div className="grid grid-rows-3 space-y-3 col-span-4 p-10 pt-5">
        <AttentionVisualization graphUrl={graphUrl} heatmapUrl={heatmapUrl} />
        <div className="row-span-2">
          <div className="text-2xl font-semibold tracking-tight mb-2">
            Result
          </div>
          <div className="p-5">
            <ProbabilityDistribution distributions={distributions} />
          </div>
        </div>
      </div>
    </FormProvider>
  );
}
