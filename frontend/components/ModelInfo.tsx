"use client";

import React, { useEffect, useState } from "react";
import axios from "axios";
import { Button } from "./ui/button";
import { Trash } from "lucide-react";
import { BACKEND_BASE_URL } from "@/app/(routes)/training/page";

type EditModelProps = {
  modelName: string;
  deleteModel: (modelName: string) => void;
};

type ModelData = {
  model_id: string;
  details: string;
};

/**
 * ModelInfo component
 *
 * This component is responsible for fetching and displaying detailed information about a specific dataset (model).
 * It also provides functionality to delete the model.
 *
 * Props:
 * - modelName: The name of the model to display information for.
 * - deleteModel: Function to handle the deletion of the model.
 *
 * State:
 * - modelData: Stores the detailed information of the model.
 * - details: Stores the parsed details of the model for display.
 *
 * Functionality:
 * - Fetches model data from the backend when the component mounts or when the modelName prop changes.
 * - Displays the model details and allows the user to delete the model.
 */
export default function ModelInfo({ modelName, deleteModel }: EditModelProps) {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [details, setDetails] = useState<string[]>([]);

  useEffect(() => {
    const fetchModelData = async () => {
      const featureResponse = await axios.get(
        `${BACKEND_BASE_URL}/dataset/data/${modelName}`
      );

      const parsedFeature = JSON.parse(featureResponse.data);

      setModelData(parsedFeature);
      setDetails(Object.values(parsedFeature.details));
    };

    fetchModelData();
  }, [modelName]);

  if (!modelData || modelName !== modelData.model_id)
    return <div>Loading...</div>;

  return (
    <>
      <div className="border rounded-md p-5">
        <h1 className="scroll-m-20 text-2xl font-extrabold tracking-tight mb-10">
          Dataset Information
        </h1>
        <p className="mb-5">
          <strong>Dataset Name:</strong> {modelName}
        </p>
        {details.map((detail, index) => (
          <div key={index} className="mb-5">
            <strong>{`Omic data ${index + 1}:`}</strong>
            <div className="mt-1 ml-5">{`Description: ${detail}`}</div>
          </div>
        ))}
      </div>
      <div className="flex justify-end mt-2">
        <Button
          variant={"destructive"}
          onClick={() => deleteModel(modelData.model_id)}
          size={"sm"}
        >
          <Trash className="size-4" />
        </Button>
      </div>
    </>
  );
}
