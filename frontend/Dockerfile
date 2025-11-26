# Use Node.js
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package files first
COPY package.json .
RUN npm install

# Copy the rest of the code
COPY . .

# Expose Vite port
EXPOSE 5173

# Run the app
# --host is CRITICAL for Docker to allow browser access
CMD ["npm", "run", "dev", "--", "--host"]