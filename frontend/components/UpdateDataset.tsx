"use client";

import React from "react";

import { Input } from "@/components/ui/input";

import { Textarea } from "@/components/ui/textarea";

import { Plus, Trash } from "lucide-react";
import { Button } from "./ui/button";

import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";

import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFieldArray, useForm } from "react-hook-form";
import axios from "axios";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { BACKEND_BASE_URL } from "@/app/(routes)/training/page";

const csvUploadObj = z.object({
  feature_name: z
    .instanceof(File, {
      message: "File must be a valid CSV file.",
    })
    .nullable(),
  feature: z
    .instanceof(File, { message: "File must be a valid CSV file." })
    .nullable(),
  details: z.string({ required_error: "Label is required." }),
});

const formSchema = z.object({
  model_name: z.string({ required_error: "Model name is required." }),
  datasets: z.array(csvUploadObj).min(1, "At least one dataset is required."),
  label: z.instanceof(File, { message: "File must be a valid CSV file." }),
});

/**
 * UpdateDataset component
 *
 * This component provides a form for updating datasets. Users can upload CSV files containing features,
 * feature names, and labels. The form allows adding multiple omic datasets and validates the input using Zod schema.
 *
 * Functionality:
 * - Validates form inputs using Zod schema and react-hook-form.
 * - Handles multiple omic dataset inputs using useFieldArray.
 * - Submits the form data to the backend server via an HTTP POST request.
 *
 * Fields:
 * - model_name: The name of the model (required).
 * - datasets: An array of objects containing feature CSV, feature name CSV, and details (at least one required).
 * - label: The label CSV file (required).
 */
export default function UpdateDataset() {
  const router = useRouter();
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      datasets: [{ feature_name: null, feature: null, details: "" }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    name: "datasets",
    control: form.control,
  });

  const addDetail = (e: any) => {
    e.preventDefault();
    append({
      feature_name: null,
      feature: null,
      details: "",
    });
  };

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const formData = new FormData();
    formData.append("model_id", values.model_name);

    values.datasets.forEach((dataset, index) => {
      if (!dataset.feature) {
        toast.error(`Omics ${index + 1} is missing Feature CSV.`);
        throw new Error(`Omics ${index + 1} is missing Feature CSV.`);
        return;
      }

      if (!dataset.feature_name) {
        toast.error(`Omics ${index + 1} is missing Feature Name CSV.`);
        throw new Error(`Omics ${index + 1} is missing Feature Name CSV.`);
        return;
      }
      formData.append(`feature_names`, dataset.feature_name);
      formData.append(`features`, dataset.feature);
      formData.append(`details`, dataset.details);
    });

    formData.append("labels", values.label);

    try {
      const response = await axios.post(
        `${BACKEND_BASE_URL}/upload-dataset/`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      router.push(`/training`);
      toast.success("Dataset uploaded successfully.");
    } catch (error: any) {
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
  }

  return (
    <>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="grid w-full items-start gap-6"
        >
          <FormField
            control={form.control}
            name={`model_name`}
            render={({ field }) => (
              <FormItem>
                <FormLabel>Model Name</FormLabel>
                <FormDescription />
                <FormControl>
                  <Input
                    placeholder="Model Name"
                    {...field}
                    className="text-sm h-10"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          {fields.map((field, index) => (
            <div key={field.id}>
              <fieldset className="grid gap-6 rounded-lg border p-4">
                <legend className="-ml-1 px-1 text-sm font-medium">
                  Omic {index + 1}
                </legend>
                <FormField
                  control={form.control}
                  name={`datasets.${index}.feature`}
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Feature CSV</FormLabel>
                      <FormDescription />
                      <FormControl>
                        <Input
                          type="file"
                          onChange={(e) => {
                            field.onChange(e.target.files?.[0] || null);
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name={`datasets.${index}.feature_name`}
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Feature Name CSV</FormLabel>
                      <FormDescription />
                      <FormControl>
                        <Input
                          type="file"
                          onChange={(e) => {
                            field.onChange(e.target.files?.[0] || null);
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name={`datasets.${index}.details`}
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Details</FormLabel>
                      <FormDescription />
                      <FormControl>
                        <Textarea
                          placeholder="Note"
                          {...field}
                          className="text-sm h-10"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </fieldset>
            </div>
          ))}

          <div className="flex justify-between">
            <Button
              variant={"outline"}
              className="flex justify-between gap-2 items-center"
              onClick={(e) => addDetail(e)}
            >
              <Plus className="size-4" />
              <div>Add Omic Data</div>
            </Button>

            {fields.length > 1 && (
              <Button
                variant="destructive"
                className="flex justify-between gap-2 items-center"
                onClick={(e) => {
                  e.preventDefault();
                  remove(fields.length - 1);
                }}
              >
                <Trash className="size-4" />
                <div>Delete Omic Data</div>
              </Button>
            )}
          </div>

          <FormField
            control={form.control}
            name={`label`}
            render={({ field }) => (
              <FormItem>
                <FormLabel>Label CSV</FormLabel>
                <FormDescription />
                <FormControl>
                  <Input
                    type="file"
                    onChange={(e) => {
                      field.onChange(e.target.files?.[0] || null);
                    }}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <div className="flex justify-between gap-2 items-center">
            <div className="flex justify-between gap-2 items-center">
              <Button type="submit">Create</Button>
            </div>
          </div>
        </form>
      </Form>
    </>
  );
}
