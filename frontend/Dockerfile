# Use an official Node.js runtime as a parent image
FROM node:20.16.0

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Copy the rest of the application code
COPY . .

# Install dependencies
RUN npm ci

# Expose port 3000 (or whichever port your frontend uses)
EXPOSE 3000

# Command to run the frontend
CMD ["npm", "run", "dev"]

