"use client";

import { ImageUploader } from '@/components/ImageUploader';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowRightIcon } from "@heroicons/react/24/outline";

export default function DashboardPage() {
  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        className="w-full max-w-4xl"
      >
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-semibold tracking-tight mb-3 font-sans" style={{letterSpacing: '-0.02em'}}>
              Welcome to <span className="bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">Dashboard</span>
            </h1>
            <p className="text-lg text-muted-foreground font-sans">
              Upload an image to analyze concrete strength
            </p>
          </div>
          <div className="space-y-8">
            <ImageUploader />
            <div className="flex justify-end">
              <Link href="/history">
                <Button
                  variant="outline"
                  size="lg"
                  className="flex items-center justify-center gap-2 text-lg font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 border-black/80 dark:border-white/10 hover:bg-black/10 dark:hover:bg-white/5 rounded-xl"
                >
                  View History <ArrowRightIcon className="w-5 h-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
