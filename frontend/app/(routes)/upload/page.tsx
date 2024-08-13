import UpdateDataset from "@/components/UpdateDataset";
import React from "react";

export default function UploadPage() {
  return (
    <>
      <div className="col-span-1 p-5">
        <div className="text-md font-semibold tracking-tight">
          Create new model
        </div>
      </div>
      <div className="col-span-5 p-10">
        <UpdateDataset />
      </div>
    </>
  );
}
