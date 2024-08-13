
# Frontend for Multi-Omics Data Integration Project

This document provides an overview and setup instructions for the frontend of the Multi-Omics Data Integration project, focusing on integrating multi-omics data using interpretable graph convolutional networks for biomarker identification.

## Project Overview

The frontend part of this project is developed using modern web technologies including React, TypeScript, and Next.js. The main goal of the frontend is to provide an intuitive interface for users to upload data, run models, and visualize results effectively.

## File Structure

- **page.tsx**: Handles the main page layout and navigation.
- **DatasetForm.tsx**: Manages the form for dataset input, including omic data details.
- **ModelInfo.tsx**: Displays model information and allows users to interact with model features.
- **ProbabilityDistribution.tsx**: Provides functionality for generating and displaying probability distributions.
- **SideNav.tsx**: Contains the side navigation menu for easy access to different sections of the platform.
- **UpdateDataset.tsx**: Facilitates the update of existing datasets with new information.
- **AttentionVisualization.tsx**: Visualizes attention mechanisms in the graph convolutional networks.
- **FeatureCustomization.tsx**: Allows customization of features used in the model.
- **HyperparametersForm.tsx**: Manages the form for setting and adjusting hyperparameters.
- **ModelSelection.tsx**: Allows users to select and compare different models.
- **ShapGeneration.tsx**: Generates SHAP (SHapley Additive exPlanations) values for model interpretability.

## Setup Instructions

To set up the frontend locally, follow these steps:

1. **Clone the repository**:
   \`\`\`bash
   git clone https://github.com/your-username/comp3900-project-frontend.git
   cd comp3900-project-frontend
   \`\`\`

2. **Install dependencies**:
   \`\`\`bash
   npm install
   \`\`\`

3. **Start the development server**:
   \`\`\`bash
   npm run dev
   \`\`\`

4. **Open the application**:
   Open your browser and navigate to `http://localhost:3000` to see the application in action.

## Usage

Once the project is set up, you can start using the frontend to interact with the multi-omics data integration model. Here are some basic usage guidelines:

- **Adding Data**: Use the `DatasetForm.tsx` to upload new datasets. Ensure all necessary details are filled out.
- **Viewing Model Information**: Navigate to the `ModelInfo.tsx` section to view detailed information about the model and interact with its features.
- **Generating Probability Distributions**: Use the `ProbabilityDistribution.tsx` component to generate and view probability distributions based on the model's predictions.
- **Updating Datasets**: If you need to update existing datasets, use the `UpdateDataset.tsx` component to add new information.
- **Navigation**: Utilize the `SideNav.tsx` for seamless navigation between different sections of the application.
- **Visualizing Attention**: Use the `AttentionVisualization.tsx` to understand how attention mechanisms are working within the graph convolutional networks.
- **Customizing Features**: Use the `FeatureCustomization.tsx` to adjust which features are used in the model.
- **Setting Hyperparameters**: Use the `HyperparametersForm.tsx` to configure and fine-tune model hyperparameters.
- **Selecting Models**: Navigate to `ModelSelection.tsx` to choose and compare different models.
- **Generating SHAP Values**: Use the `ShapGeneration.tsx` to produce SHAP values for interpreting the model's decisions.

## Contributing

Contributions to the frontend are welcome. If you would like to contribute, please follow these steps:

1. **Fork the repository**:
   Click the "Fork" button at the top right corner of the repository page.

2. **Clone the forked repository**:
   \`\`\`bash
   git clone https://github.com/your-username/comp3900-project-frontend.git
   cd comp3900-project-frontend
   \`\`\`

3. **Create a new branch**:
   \`\`\`bash
   git checkout -b feature/your-feature-name
   \`\`\`

4. **Make your changes**:
   Implement your feature or bug fix.

5. **Commit your changes**:
   \`\`\`bash
   git commit -m "Add feature: your-feature-name"
   \`\`\`

6. **Push your changes**:
   \`\`\`bash
   git push origin feature/your-feature-name
   \`\`\`

7. **Create a pull request**:
   Open a pull request from your forked repository on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
