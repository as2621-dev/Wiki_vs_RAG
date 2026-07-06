import type { Config } from "tailwindcss";

// The three fixed system hues + neutrals come from reference/design-language.md.
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        system: {
          plain_llm: "#2563eb",
          rag: "#0d9488",
          wiki: "#d97706",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
