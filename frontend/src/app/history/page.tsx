"use client"

import { History } from '@/components/History';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowLeftIcon } from "@heroicons/react/24/outline";

export default function HistoryPage() {
  return (
    <div className="flex items-center justify-center py-8">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        className="w-full max-w-4xl"
      >
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-4xl font-semibold tracking-tight mb-2 font-sans" style={{letterSpacing: '-0.02em'}}>
                Analysis <span className="bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">History</span>
              </h1>
              <p className="text-lg text-muted-foreground font-sans">
                View your previous concrete strength analyses
              </p>
            </div>
            <Link href="/dashboard">
              <Button
                variant="outline"
                size="lg"
                className="flex items-center justify-center gap-2 text-lg font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 border-black/80 dark:border-white/10 hover:bg-black/10 dark:hover:bg-white/5 rounded-xl"
              >
                <ArrowLeftIcon className="w-5 h-5" /> Back to Dashboard
              </Button>
            </Link>
          </div>
          <History />
        </div>
      </motion.div>
    </div>
  );
}
