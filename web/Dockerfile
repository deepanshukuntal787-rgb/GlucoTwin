# Stage 1: Build React/Vite application
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy sources and build production bundle
COPY . .
ENV VITE_USE_MOCK=false
ENV VITE_API_URL=/api/v1
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
