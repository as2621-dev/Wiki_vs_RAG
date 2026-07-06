import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wiki vs RAG — scaling figures",
  description: "Accuracy crossover and cost curve across corpus token count.",
};

export default function RootLayout({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
