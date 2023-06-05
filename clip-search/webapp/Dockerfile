FROM node:18

COPY package.json ./package.json
RUN yarn

COPY . .

# Build the app
RUN yarn build

# Run the app
CMD ["yarn", "start"]

