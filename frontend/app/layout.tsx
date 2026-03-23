import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Car Damage Severity Predictor",
  description: "Upload a car damage photo to get an AI severity assessment.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen text-gray-900 antialiased">{children}</body>
    </html>
  );
}
