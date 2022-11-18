const defaultTheme = require("tailwindcss/defaultTheme");

module.exports = {
  content: [
    "./components/**/*.{html,js,ts,jsx,tsx}",
    "./pages/**/*.{html,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
      },
      spacing: {
        "1/2": "50%",
        "2/3": "66.666667%",
        "3/4": "75%",
        "4/5": "80%",
      },
    },
  },
};
