"use client";

import React, { useEffect, useState } from "react";
import axios from "axios";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import ModelInfo from "./ModelInfo";
import { BACKEND_BASE_URL } from "@/app/(routes)/training/page";

/**
 * DatasetForm component
 *
 * This component is responsible for fetching and displaying a list of datasets (models) from the backend.
 * It allows users to select a dataset to view its details and delete a dataset if needed.
 *
 * Functionality:
 * - Fetches dataset list from the backend when the component mounts.
 * - Displays the list of datasets with buttons to select each dataset.
 * - Displays detailed information about the selected dataset using the ModelInfo component.
 * - Allows users to delete a dataset, which triggers a backend request and updates the list.
 */
export default function DatasetForm() {
  const [models, setModels] = useState([]); // State to store the list of models.
  const [selectedModel, setSelectedModel] = useState(""); // State to store the currently selected model.

  // useEffect hook to fetch models from the backend when the component mounts.
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

  const deleteModel = async (modelName: any) => {
    try {
      await axios.delete(`${BACKEND_BASE_URL}/dataset/${modelName}`);

      setModels(models.filter((model) => model !== modelName));

      // refresh the current window
      window.location.reload();
    } catch (error) {
      console.error("Error deleting model:", error);
    }
  };

  return (
    <>
      <div className="relative flex flex-col items-start gap-5 w-full p-5 col-span-2">
        <div className="text-md font-semibold tracking-tight mb-3">
          Your Dataset
        </div>
        <div className="w-full flex flex-col gap-2">
          {models.length === 0 && (
            <div className="text-sm font-light text-zinc-600 tracking-tight">
              No models found
            </div>
          )}
          {models.map((model) => (
            <div key={model} className="flex justify-between w-full gap-2">
              <Button
                variant={"outline"}
                className={cn(
                  model === selectedModel
                    ? "bg-muted hover:bg-muted"
                    : "hover:bg-transparent hover:underline",
                  "justify-start border-none w-full"
                )}
                onClick={() => setSelectedModel(model)}
              >
                {model}
              </Button>
            </div>
          ))}
        </div>
      </div>
      <div className="relative flex h-full min-h-[50vh] flex-col rounded-xl col-span-4 p-10 pt-20">
        {selectedModel && (
          <ModelInfo modelName={selectedModel} deleteModel={deleteModel} />
        )}
      </div>
    </>
  );
}
