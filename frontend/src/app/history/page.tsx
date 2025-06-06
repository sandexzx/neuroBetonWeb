"use client"

import { History } from '@/components/History';
import { motion } from 'framer-motion';

export default function HistoryPage() {
  return (
    <div className="h-full flex flex-col">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        className="w-full max-w-6xl mx-auto px-8 flex-shrink-0"
      >
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8 mb-6">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-semibold tracking-tight mb-2 font-sans" style={{letterSpacing: '-0.02em'}}>
              Analysis <span className="bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">History</span>
            </h1>
            <p className="text-lg text-muted-foreground font-sans">
              View your previous concrete strength analyses
            </p>
          </div>
        </div>
      </motion.div>
      
      <div className="flex-1 w-full max-w-6xl mx-auto px-8 min-h-0">
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm rounded-2xl shadow-2xl h-full overflow-hidden">
          <History />
        </div>
      </div>
    </div>
  );
}
