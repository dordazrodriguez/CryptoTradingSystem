# Frontend Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Build-time args for React envs (must start with REACT_APP_)
ARG REACT_APP_API_URL
ARG REACT_APP_ALPACA_API_KEY
ARG REACT_APP_ALPACA_SECRET_KEY
ARG REACT_APP_PAPER_TRADING

# Make them available during build
ENV REACT_APP_API_URL=$REACT_APP_API_URL \
    REACT_APP_ALPACA_API_KEY=$REACT_APP_ALPACA_API_KEY \
    REACT_APP_ALPACA_SECRET_KEY=$REACT_APP_ALPACA_SECRET_KEY \
    REACT_APP_PAPER_TRADING=$REACT_APP_PAPER_TRADING

# Copy package files
COPY frontend/package*.json ./

# Install dependencies (tolerate legacy peer deps due to old material-ui)
RUN npm install --omit=dev --legacy-peer-deps

# Copy source code
COPY frontend/ .

# Build the application
RUN npm run build

# Install serve to run the production build
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Health check using Node.js (no extra dependencies needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (r) => { process.exit(r.statusCode === 200 ? 0 : 1) }).on('error', () => process.exit(1))"

# Run the application
CMD ["serve", "-s", "build", "-l", "3000"]
