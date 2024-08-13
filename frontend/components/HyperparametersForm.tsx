import React from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
} from "@/components/ui/form";
import ModelSelection from "./ModelSelection";

/**
 * HyperparametersForm component
 *
 * This component provides a form for users to input the name of a model and its hyperparameters.
 * It includes validation using Zod schema and React Hook Form, and displays appropriate error messages.
 *
 * Props:
 * - onSubmit: Function to handle form submission with form values.
 * - isTraining: Boolean indicating whether the training process is ongoing.
 * - setSelectedModel: Function to set the selected model.
 * - models: Array of available models for selection.
 */
const formSchema = z.object({
  model_name: z.string().nonempty("Model name is required."),
  hyperparameters: z
    .string()
    .nonempty("Hyperparameters must be a positive number."),
});

type FormValues = z.infer<typeof formSchema>;

interface HyperparametersFormProps {
  onSubmit: (values: FormValues) => void;
  isTraining: boolean;
  setSelectedModel: (model: string) => void;
  models: string[];
}

const HyperparametersForm: React.FC<HyperparametersFormProps> = ({
  onSubmit,
  isTraining,
  setSelectedModel,
  models,
}) => {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
  });

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="grid w-full items-start gap-6"
      >
        <FormField
          control={form.control}
          name="model_name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Model Name</FormLabel>
              <FormControl>
                <ModelSelection
                  setSelectedModel={(model) => {
                    field.onChange(model);
                    setSelectedModel(model);
                  }}
                  models={models}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="hyperparameters"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Hyperparameter</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  placeholder="avg num of edges per node retained"
                  {...field}
                  className="text-sm h-10"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit" disabled={isTraining}>
          {isTraining ? "Training..." : "Train"}
        </Button>
      </form>
    </Form>
  );
};

export default HyperparametersForm;
