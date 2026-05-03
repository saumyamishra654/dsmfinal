import type { Metadata } from "next";
import { Source_Serif_4, Playfair_Display } from "next/font/google";
import "./globals.css";
import ReportNav from "@/components/ReportNav";

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-body",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-heading",
});

export const metadata: Metadata = {
  title: "India's Digital Transformation | DSM Research Report",
  description:
    "A comprehensive empirical analysis of India's telecommunications revolution, digital payments ecosystem, and their socioeconomic impacts (2009-2023).",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sourceSerif.variable} ${playfair.variable} h-full antialiased`}>
      <body className="min-h-full flex">
        <ReportNav />
        <main className="flex-1 ml-64 min-h-screen overflow-y-auto bg-[#f5e6d3]">
          <div className="max-w-5xl mx-auto px-8 py-10">{children}</div>
        </main>
      </body>
    </html>
  );
}
