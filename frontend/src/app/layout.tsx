import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Header from "../components/Header";
import { AuthProvider } from "@/lib/auth";
import { ToastProvider } from "@/components/ui/toast";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NeuroBeton",
  description: "Neural network analysis for concrete strength",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <AuthProvider>
          <ToastProvider>
            <div className="h-screen flex flex-col overflow-hidden">
              <Header />
              <main className="flex-1 container mx-auto p-4 overflow-y-auto">
                {children}
              </main>
              <footer className="bg-primary/10 p-2 flex-shrink-0">
                <div className="container mx-auto text-muted-foreground">
                  <p className="text-center">© 2025 NeuroBeton. All rights reserved.</p>
                </div>
              </footer>
            </div>
          </ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
