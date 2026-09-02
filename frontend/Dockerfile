# Frontend container — serves the built React app.
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
RUN npm install -g vite
COPY --from=build /app/dist ./dist
EXPOSE 5173
CMD ["vite", "preview", "--host", "0.0.0.0", "--port", "5173"]
